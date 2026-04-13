import numpy as np
import gymnasium as gym
from tdmpc2.envs.wrappers.timeout import Timeout
import torch
from collections import defaultdict, deque

from isaaclab.app import AppLauncher


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
        return obs, info

    def step(self, action):
        action = torch.from_numpy(action)
        obs, reward, terminated, truncated, info = self.env.step(action)

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
            obs,
            reward,
            terminated,
            truncated,
            info,
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
        self.observation_space = env.observation_space
        assert "rgb" in self.observation_space.keys()
        previous_shape = self.observation_space["rgb"].shape
        self.observation_space["rgb"] = gym.spaces.Box(
            low=0, high=255, shape=(num_frames*previous_shape[0], *previous_shape[1:]), dtype=np.uint8)
        self._frames = deque([], maxlen=num_frames)

    def _get_visual_obs(self, obs, is_reset=False):
        frame = obs["rgb"]
        num_frames = self._frames.maxlen if is_reset else 1
        for _ in range(num_frames):
            self._frames.append(frame)
        past_n_frames = torch.concatenate(tuple(self._frames), axis=1)
        obs["rgb"] = past_n_frames
        return obs

    def reset(self, env_id=None):
        obs, info = self.env.reset(env_id=env_id)
        return self._get_visual_obs(obs, is_reset=True), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._get_visual_obs(obs), reward, terminated, truncated, info

app_launcher = None
simulation_app = None
tasks_module = None
def make_env(cfg):
    """
    Make classic/MuJoCo environment.
    """
    print("ENTERED make_env")
    # Instantiate the IsaacLab simulator
    global app_launcher, simulation_app, tasks_module
    app_launcher = AppLauncher(launcher_args={
        "livestream": int(cfg.visualize),
        "enable_cameras": cfg.enable_cameras,
        "device": cfg.device
    })
    simulation_app = app_launcher.app
    import custom_isaaclab_tasks.tasks
    tasks_module = custom_isaaclab_tasks.tasks

    # Load the env_cfg
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
    # Update the env_cfg based on the tdmpc2_cfg
    env_cfg = load_cfg_from_registry(cfg.task, "env_cfg_entry_point")
    env_cfg.scene.num_envs = cfg.num_envs
    env_cfg.device = cfg.device
    env_cfg.sim.device = cfg.device
    env_cfg.task_index = cfg.task_index
    env_cfg.seed = cfg.seed

    env = gym.make(cfg.task, cfg=env_cfg, render_mode="rgb_array")
    env = IsaacLabWrapper(env, cfg, cfg.task)
    if "rgb" in cfg.obs:
        env = Pixels(cfg=cfg, env=env)
    return env
