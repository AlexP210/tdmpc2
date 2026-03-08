import numpy as np
import gymnasium as gym
from dstl.envs.wrappers.timeout import Timeout
import torch
import evaluation.tasks  # noqa: F401
from collections import defaultdict, deque

class IsaacLabWrapper(gym.Wrapper):
    def __init__(self, env, cfg, task_name):
        super().__init__(env)
        self.env = env
        self.cfg = cfg
        self.task_name = task_name

    def reset(self, env_id=None):
        if env_id is None:
            obs, info = self.env.reset()
        else:
            self.env.unwrapped._reset_idx(
                env_ids=torch.tensor(
                    [
                        env_id,
                    ]
                )
            )
            obs = self.env.unwrapped._get_observations()
        if type(obs) == type(dict()):
            obs = obs[self.cfg.obs]
        return self.obs_to_cpu(obs), self.info_to_cpu(info)

    def step(self, action):
        action = torch.from_numpy(action)
        obs, reward, terminated, truncated, info = self.env.step(action)
         # If the env is returning us a dict
        if type(obs) == type(dict()):
            obs = obs[self.cfg.obs]

        if "Manager" in self.task_name:
            # For the manager-based envs, we need to get the termination and truncation
            # signal from the `info` dict; if we do it normally, then the ManagerBasedRLEnv
            # will automatically reset specific envs that have the termination or truncated flags
            # For consistency with the FactoryEnv's, we want to reset them all together instead so 
            # can't do this
            terminated = torch.full_like(terminated, fill_value=False)
            truncated = torch.full_like(truncated, fill_value=False)
            for key, val in info.items():
                if key.split("/")[0] == "termination": terminated |= val
                elif key.split("/")[0] == "truncation": truncated |= val
            # Shuffle around the info dict
            successes = info["successes"]
            episode_length = info["episode_lengths"]
            info = {}
            info["successes"] = successes
            info["episode_lengths"] = episode_length
        return_value = (
            self.obs_to_cpu(obs),
            reward.cpu(),
            terminated.cpu(),
            truncated.cpu(),
            self.info_to_cpu(info),
        )
        return return_value

    def render(self):
        return self.env.render()

    def info_to_cpu(self, info):
        return {key: val.cpu()  for key, val in info.items()}

    def obs_to_cpu(self, obs):
        return obs.cpu()

    @property
    def unwrapped(self):
        return self.env.unwrapped

    def render(self, **kwargs):
        return self.env.render(**kwargs)

    def _get_obs(self, is_reset=False):
        return

class Pixels(gym.Wrapper):
    def __init__(self, env, cfg, num_frames=3):
        super().__init__(env)
        self.cfg = cfg
        self.env = env
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(num_frames*3, 64, 64), dtype=np.uint8)
        self._frames = deque([], maxlen=num_frames)

    def _get_visual_obs(self, frame, is_reset=False):
        num_frames = self._frames.maxlen if is_reset else 1
        for _ in range(num_frames):
            self._frames.append(frame)
        past_n_frames = torch.concatenate(tuple(self._frames), axis=1)
        return past_n_frames

    def reset(self, env_id=None):
        obs, info = self.env.reset(env_id=env_id)
        return self._get_visual_obs(obs, is_reset=True), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._get_visual_obs(obs), reward, terminated, truncated, info


def make_env(cfg, env_cfg):
    """
    Make classic/MuJoCo environment.
    """
    env = gym.make(cfg.task, cfg=env_cfg, render_mode="rgb_array")
    env = IsaacLabWrapper(env, cfg, cfg.task)
    if cfg.obs == "rgb":
        env = Pixels(cfg=cfg, env=env)
    return env
