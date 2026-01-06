import numpy as np
import gymnasium as gym
from tdmpc2.envs.wrappers.timeout import Timeout
import torch
import evaluation.tasks  # noqa: F401

ISAACLAB_TASKS = {
    "Template-Evaluation-Direct-v0": "Template-Evaluation-Direct-v0",
    "Isaac-Forge-PegInsert-Direct-v0": "Isaac-Forge-PegInsert-Direct-v0",
    "Isaac-Forge-GearMesh-Direct-v0": "Isaac-Forge-GearMesh-Direct-v0",
    "Isaac-Forge-NutThread-Direct-v0": "Isaac-Forge-NutThread-Direct-v0",
    "DistillPlan-boxplace-v0": "DistillPlan-boxplace-v0",
    "DistillPlan-Place-Toy2Box-Agibot-Right-Arm-RmpFlow-v0": "DistillPlan-Place-Toy2Box-Agibot-Right-Arm-RmpFlow-v0",
    "BoxPlace-Direct-v0": "BoxPlace-Direct-v0",
    "DistillPlan-Forge-PegInsert-Direct-v0": "DistillPlan-Forge-PegInsert-Direct-v0",
    "Locomotion-Manager-v0": "Locomotion-Manager-v0",
}


class FlattenAction(gym.ActionWrapper):
    """Action wrapper that flattens the action."""

    def __init__(self, env):
        super(FlattenAction, self).__init__(env)
        self.action_space = gym.spaces.utils.flatten_space(self.env.action_space)

    def action(self, action):
        return gym.spaces.utils.unflatten(self.env.action_space, action)

    def reverse_action(self, action):
        return gym.spaces.utils.flatten(self.env.action_space, action)


class IsaacLabWrapper(gym.Wrapper):
    def __init__(self, env, cfg, task_name):
        super().__init__(env)
        self.env = env
        self.cfg = cfg
        self.task_name = task_name
        self._cumulative_reward = 0

    def reset(self, **kwargs):
        self._cumulative_reward = 0
        obs, info = self.env.reset()
        return self._squeeze_obs(obs), info

    def step(self, action):
        action = torch.from_numpy(action)
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._cumulative_reward += reward
        done = (terminated | truncated).all()
        info["terminated"] = done
        return_value = (
            self._squeeze_obs(obs),
            reward.cpu(),
            terminated.cpu(),
            truncated.cpu(),
            info,
        )
        del obs, action, terminated, truncated, info
        return return_value

    def render(self):
        return self.env.render()

    def _squeeze_obs(self, obs):
        if self.task_name == "Locomotion-Manager-v0":
            new_obs = {k: o.cpu() for k, o in obs.items()}  #
        if self.task_name == "BoxPlace-Direct-v0":
            new_obs = obs["policy"].cpu()  # .squeeze()
        # print(new_obs)
        # print(new_obs.shape)
        return new_obs

    @property
    def unwrapped(self):
        return self.env.unwrapped

    def render(self, **kwargs):
        return self.env.render(**kwargs)

    def _get_obs(self, is_reset=False):
        return


def make_env(cfg, env_cfg):
    """
    Make classic/MuJoCo environment.
    """
    print("In IsaacLab env maker")
    if not cfg.task in ISAACLAB_TASKS:
        raise ValueError("Unknown task:", cfg.task)
    env = gym.make(ISAACLAB_TASKS[cfg.task], cfg=env_cfg, render_mode="rgb_array")
    env = IsaacLabWrapper(env, cfg, cfg.task)
    # env = gym.wrappers.FlattenObservation(env)
    # env = FlattenAction(env)
    # env = Timeout(env, max_episode_steps=600)
    return env
