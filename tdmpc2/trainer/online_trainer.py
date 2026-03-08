from time import time

import numpy as np
import torch
import h5py
import os
from tensordict.tensordict import TensorDict
from dstl.trainer.base import Trainer


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

        # Run all environments `self.cfg_eval_episodes` times
        for i in range(self.cfg.eval_episodes):
            (parallel_obs, parallel_info), parallel_done, parallel_ep_reward, t = (
                self.env.reset(),
                torch.full(size=(self.cfg.num_envs,1),fill_value=False),
                torch.full(size=(self.cfg.num_envs,1),fill_value=0.0),
                0,
            )
            if self.cfg.save_video:
                self.logger.video.init(self.env, enabled=(i == 0))

            while not parallel_done.all():
                # Find which ones are not done yet
                not_done_envs = torch.nonzero(~parallel_done, as_tuple=True)[0].tolist()
                torch.compiler.cudagraph_mark_step_begin()

                # Compute the actions for those that are not done yet
                parallel_actions = torch.zeros(
                    size=(self.cfg.num_envs, self.cfg.action_dim)
                )
                for env_id in not_done_envs:
                    parallel_actions[env_id], _ = self.agent.act(parallel_obs[env_id], t0=t == 0, eval_mode=True)
                
                # Apply the actions for all of them (only non-zero for those that are not done)
                (
                    parallel_obs,
                    parallel_reward,
                    parallel_terminated,
                    parallel_truncated,
                    info,
                ) = self.env.step(parallel_actions)

                # For each env that was just updated, add the reward
                for env_id in not_done_envs:
                    parallel_ep_reward[env_id] += parallel_reward[env_id]
                    t += 1

                # Record a frame of video
                if self.cfg.save_video:
                    self.logger.video.record(self.env)

                # Find which envs are done
                parallel_done = parallel_terminated | parallel_truncated
            # Episode is done, average the total reward, episode success and episode length across all envs
            ep_rewards.append(parallel_ep_reward.mean())
            ep_successes.append(info["successes"].float().mean())
            ep_lengths.append(info["episode_lengths"].float().mean())

            # Save the video
            if self.cfg.save_video:
                self.logger.video.save(self._step)

        return dict(
            episode_reward=np.nanmean(ep_rewards),
            episode_success=np.nanmean(ep_successes),
            episode_length=np.nanmean(ep_lengths),
        )

    def to_td(self, obs, state=None, action=None, reward=None, terminated=None):
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
        if state is None:
            terminated = torch.tensor(float("nan"))
        td = TensorDict(
            obs=obs,
            state=state.unsqueeze(0),
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
        # self._holding_envs = set()  # Envs currently holding while we wait for reset
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
        self._step_at_last_checkpoint = 0


        # While we need more steps
        while self._step <= self.cfg.steps:
            # Evaluate agent periodically
            if self._step - self._step_at_last_eval >= self.cfg.eval_freq:
                eval_next = True  # Eval at the next opportunity

            # If all environments are in a `done` state, process the episode data and reset them
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
                    self._holding_envs = set()

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
                obs, info = self.env.reset()
                state = info["state"]
                for env_id in range(self.cfg.num_envs):
                    first_obs = obs[env_id]
                    first_state = state[env_id]
                    self._tds_for_each_env[env_id].append(self.to_td(first_obs, first_state))

            # Which environments are still running
            not_done_envs = torch.nonzero(~done, as_tuple=True)[0].tolist()

            # Store the transitions
            transitions = {}

            # Prepare actions for the envs that are not done
            actions = torch.zeros(size=(self.cfg.num_envs, self.cfg.action_dim))
            for env_id in not_done_envs:
                last_obs = self._tds_for_each_env[env_id][-1]["obs"][0]
                last_state = self._tds_for_each_env[env_id][-1]["state"][0]
                # Choose a random or planned action depending on whether we're still collecting seed data
                if self._pretraining_done:
                    a, _ = self.agent.act(
                        last_obs, t0=len(self._tds_for_each_env[env_id]) == 1
                    )
                else:
                    a = self.env.rand_act()[env_id]
                actions[env_id] = a
                
                # If we're saving the training data, then store the first half of the transition (z,a) for each env
                if self.logger.save_training_data > 0:
                    transitions[env_id] = [last_obs, a, last_state]

            # Step the envs with the actions
            obs, reward, terminated, truncated, info = self.env.step(actions)
            state = info["state"]
            done = terminated | truncated

            # Turn the transition to a tensordict and add to the transition list
            for env_id in not_done_envs:
                self._tds_for_each_env[env_id].append(
                    self.to_td(
                        obs[env_id], state[env_id], actions[env_id], reward[env_id], terminated[env_id]
                    )
                )
                self._step += 1

                # If we're saving training data, then log the transition
                if self.logger.save_training_data > 0:
                    o, a, s = transitions[env_id]
                    self.logger.log_transition(o, a, reward[env_id], obs[env_id], terminated[env_id], truncated[env_id], s, state[env_id])

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

            # Save a checkpoint if it's time to
            if self._step - self._step_at_last_checkpoint > self.cfg.checkpoint_interval:
                self.logger.save_agent(self.agent, identifier=f"{self._step}")
                self._step_at_last_checkpoint = self._step

        self.logger.finish(self.agent)
