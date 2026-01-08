from time import time

import numpy as np
import torch
from tensordict.tensordict import TensorDict
from tdmpc2.trainer.base import Trainer


class OnlineTrainer(Trainer):
    """Trainer class for single-task online TD-MPC2 training."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._step = 0
        self._ep_idx = 0
        self._start_time = time()

    def common_metrics(self):
        """Return a dictionary of current metrics."""
        elapsed_time = time() - self._start_time
        return dict(
            step=self._step,
            episode=self._ep_idx,
            elapsed_time=elapsed_time,
            steps_per_second=self._step / elapsed_time,
        )

    def eval(self):
        """Evaluate a TD-MPC2 agent."""
        # Store the total reward, episode success, episode length for each of the eval episodes
        ep_rewards, ep_successes, ep_lengths = [], [], []

        # Run all `num_envs` envronments until we've collected enough episodes
        for i in range(self.cfg.eval_episodes):
            parallel_obs, done, ep_reward, t = (
                self.env.reset(),
                False,
                0,
                0,
            )
            obs = parallel_obs[0]
            if self.cfg.save_video:
                self.logger.video.init(self.env, enabled=(i == 0))
            while not done:
                torch.compiler.cudagraph_mark_step_begin()
                parallel_actions = torch.zeros(
                    size=(self.cfg.num_envs, self.cfg.action_dim)
                )
                parallel_actions[0] = self.agent.act(obs, t0=t == 0, eval_mode=True)
                (
                    parallel_obs,
                    parallel_reward,
                    parallel_terminated,
                    parallel_truncated,
                    info,
                ) = self.env.step(parallel_actions)
                obs = parallel_obs[0]
                reward = parallel_reward[0]
                terminated = parallel_terminated[0]
                truncated = parallel_truncated[0]
                done = terminated or truncated
                ep_reward += reward
                t += 1
                if self.cfg.save_video:
                    self.logger.video.record(self.env)

            ep_rewards.append(ep_reward)
            ep_successes.append(info["successes"][0])
            ep_lengths.append(info["episode_lengths"][0])

            if self.cfg.save_video:
                self.logger.video.save(self._step)

        return dict(
            episode_reward=np.nanmean(ep_rewards),
            episode_success=np.nanmean(ep_successes),
            episode_length=np.nanmean(ep_lengths),
        )

    def to_td(self, obs, action=None, reward=None, terminated=None):
        """Creates a TensorDict for a new episode."""
        if isinstance(obs, dict):
            obs = TensorDict(obs, batch_size=(), device="cpu")
        else:
            obs = obs.unsqueeze(0).cpu()
        if action is None:
            action = torch.full_like(self.env.rand_act()[0], float("nan"))
        if reward is None:
            reward = torch.tensor(float("nan"))
        if terminated is None:
            terminated = torch.tensor(float("nan"))
        td = TensorDict(
            obs=obs,
            action=action.unsqueeze(0),
            reward=reward.unsqueeze(0),
            terminated=terminated.unsqueeze(0),
            batch_size=(1,),
        )
        return td

    def train(self):
        """Train a TD-MPC2 agent."""
        train_metrics = {}  # Dict for the env and agent metrics after each episode
        done = torch.full(
            (self.cfg.num_envs, 1), True
        )  # Ready-for-reset flag for each env
        eval_next = False  # Waiting-for-eval flag
        self._tds_for_each_env = [
            [] for _ in range(self.cfg.num_envs)
        ]  # Episode history for each env
        self._holding_envs = {}  # Envs currently holding while we wait for reset
        # NOTE: Ideally we would reset envs whenever they are finished; but for some reason
        # IsaacLab's FactoryEnv examples (which box-place is based on) require all envs to be
        # reset at once. They don't explain why, but if you try to do staggered resets, IsaacSim 
        # errors out with uninformative error message "cannot set body poses in backend"
        self._pretraining_done = False # Flag to indicate if the pretraining on the seed steps has been done
        # NOTE: Had to add this because with lots of envs goign at once we can get collect > `self.cfg.seed_steps`
        # before the first episode even finishes
        self._step_at_last_eval = -float("inf")
        # NOTE: Same thing here; in between iterations of the loop self._step can and does increase by more than one
        # with multiple envs

        # While we need more steps
        while self._step <= self.cfg.steps:
            # Evaluate agent periodically
            if self._step - self._step_at_last_eval >= self.cfg.eval_freq:
                eval_next = True  # Eval at the next opportunity

            # If all environments are in a `done` state, process thee episode data and reset them
            if done.all():

                # First, if we're due for an eval, do it
                if eval_next:
                    eval_metrics = (
                        self.eval()
                    )  # Average reward, success and length for eval episodes (on env 0)
                    eval_metrics.update(
                        self.common_metrics()
                    )  # step #, episode #, elapsed time, steps / s
                    self.logger.log(eval_metrics, "eval")
                    eval_next = False
                    self._step_at_last_eval = self._step
                    # No envs are being held anymore
                    self._holding_envs = {}

                # Next, update our train metrics and add the episodes to the buffer
                if self._step > 0:
                    # Update the train metrics
                    total_reward = 0
                    total_success = 0
                    total_length = 0
                    total_terminated = 0
                    for env_id in range(self.cfg.num_envs):
                        if terminated[env_id] and not self.cfg.episodic:
                            raise ValueError(
                                "Termination detected but you are not in episodic mode. "
                                "Set `episodic=true` to enable support for terminations."
                            )
                        total_reward += torch.tensor([td["reward"] for td in self._tds_for_each_env[env_id][1:]]).sum()
                        total_success += info["successes"][env_id]
                        total_length += len(self._tds_for_each_env[env_id])
                        total_terminated += terminated[env_id]
                        # Concatenate all transitions, add to the buffer
                        self._ep_idx = self.buffer.add(
                            torch.cat(self._tds_for_each_env[env_id])
                        ) 
                        self._tds_for_each_env[env_id] = []
                    # Add the average metrics across all envs to the train metrics
                    train_metrics.update(
                        episode_reward=total_reward/self.cfg.num_envs,  # Total rewards from the episode
                        episode_success=total_success/self.cfg.num_envs,  # Whether the last transition was a success
                        episode_length=total_length/self.cfg.num_envs,  # Number of steps in the episode
                        episode_terminated=total_terminated/self.cfg.num_envs,  # Whether the last transition was a termination (as opposed to a truncation)
                    )
                    train_metrics.update(
                        self.common_metrics()
                    )  # step #, episode #, elapsed time, steps / s
                    self.logger.log(train_metrics, "train")

                # Reset the environments
                obs = self.env.reset()
                for env_id in range(self.cfg.num_envs):
                    first_obs = obs[env_id]
                    self._tds_for_each_env[env_id].append(self.to_td(first_obs))

            # If not all environments are done, track which are so we can avoid collecting further data
            # until we can do the reset
            elif done.any():
                self._holding_envs.update(torch.nonzero(done, as_tuple=True)[0]) 

            # Which environments are still running
            non_held_envs = [
                i for i in range(self.cfg.num_envs) if i not in self._holding_envs
            ]
            actions = torch.zeros(size=(self.cfg.num_envs, self.cfg.action_dim))
            for env_id in non_held_envs:
                last_obs = self._tds_for_each_env[env_id][-1]["obs"][0]
                # Choose a random or planned action depending on whether we're still collecting seed data
                if self._pretraining_done:
                    actions[env_id] = self.agent.act(
                        last_obs, t0=len(self._tds_for_each_env[env_id]) == 1
                    )
                else:
                    actions[env_id] = self.env.rand_act()[env_id]

            # Step the envs with the actions
            obs, reward, terminated, truncated, info = self.env.step(actions)
            done = terminated | truncated

            # Turn the transition to a tensordict and add to the transition list
            for env_id in non_held_envs:
                self._tds_for_each_env[env_id].append(
                    self.to_td(
                        obs[env_id], actions[env_id], reward[env_id], terminated[env_id]
                    )
                )
                self._step += 1

            # Update agent if we've collected enough for pretraining 
            # and have at least one episode in the buffer
            if self._step >= self.cfg.seed_steps and self.buffer.num_eps > 0:
                
                if not self._pretraining_done:
                    num_updates = self.cfg.seed_steps
                    print("Pretraining agent on seed data...")
                    self._pretraining_done = True
                else:
                    num_updates = self.cfg.num_envs

                for _ in range(num_updates):
                    _train_metrics = self.agent.update(
                        self.buffer
                    )  # Consistency, reward, termination, value loss + policy loss and entropy
                train_metrics.update(_train_metrics)

        self.logger.finish(self.agent)
