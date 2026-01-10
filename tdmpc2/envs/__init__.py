from copy import deepcopy
import warnings

import gymnasium as gym

from tdmpc2.envs.wrappers.multitask import MultitaskWrapper
from tdmpc2.envs.wrappers.tensor import TensorWrapper


def missing_dependencies(task):
    raise ValueError(
        f"Missing dependencies for task {task}; install dependencies to use this environment."
    )


try:
    from tdmpc2.envs.dmcontrol import make_env as make_dm_control_env
except:
    make_dm_control_env = missing_dependencies
try:
    from envs.maniskill import make_env as make_maniskill_env
except:
    make_maniskill_env = missing_dependencies
try:
    from envs.metaworld import make_env as make_metaworld_env
except:
    make_metaworld_env = missing_dependencies
try:
    from envs.myosuite import make_env as make_myosuite_env
except:
    make_myosuite_env = missing_dependencies
try:
    from envs.mujoco import make_env as make_mujoco_env
except:
    make_mujoco_env = missing_dependencies
try:
    from tdmpc2.envs.isaaclab import make_env as make_isaaclab_env
except:
    make_isaaclab_env = missing_dependencies

warnings.filterwarnings("ignore", category=DeprecationWarning)


def make_multitask_env(cfg):
    """
    Make a multi-task environment for TD-MPC2 experiments.
    """
    print("Creating multi-task environment with tasks:", cfg.tasks)
    envs = []
    for task in cfg.tasks:
        _cfg = deepcopy(cfg)
        _cfg.task = task
        _cfg.multitask = False
        env = make_env(_cfg)
        if env is None:
            raise ValueError("Unknown task:", task)
        envs.append(env)
    env = MultitaskWrapper(cfg, envs)
    cfg.obs_shapes = env._obs_dims
    cfg.action_dims = env._action_dims
    cfg.episode_lengths = env._episode_lengths
    return env


def make_env(cfg, env_cfg=None):
    """
    Make an environment for TD-MPC2 experiments.
    """
    # gym.logger.set_level(40)
    if cfg.multitask:
        env = make_multitask_env(cfg)

    else:
        env = make_isaaclab_env(cfg, env_cfg)
    # If env observation space is a Dict, return a dict with just {`key`:`shape`}
    obs_format = cfg.get("obs")
    if type(env.observation_space) == type(gym.spaces.Dict(spaces={})):
        cfg.obs_shape = {obs_format: env.observation_space.spaces[obs_format].shape}
    # If env observation space is a Box, return a dict with just {`obs_type`:`shape`}
    elif type(env.observation_space) == type(gym.spaces.Box(low=0, high=1)):
        cfg.obs_shape = {obs_format: env.observation_space.shape}
    cfg.action_dim = env.action_space.shape[1]
    cfg.episode_length = env.unwrapped.max_episode_length
    cfg.seed_steps = max(1000, 5 * cfg.episode_length)
    cfg.num_envs = env.unwrapped.num_envs
    env = TensorWrapper(env)
    return env
