import numpy as np
import gymnasium as gym
from tdmpc2.envs.wrappers.timeout import Timeout
import torch
import evaluation.tasks  # noqa: F401

ISAACLAB_TASKS = {
	"Template-Evaluation-Direct-v0":"Template-Evaluation-Direct-v0",
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
	def __init__(self, env, cfg):
		super().__init__(env)
		self.env = env
		self.cfg = cfg
		self._cumulative_reward = 0

	def reset(self, **kwargs):
		self._cumulative_reward = 0
		obs, info = self.env.reset()
		return self._squeeze_obs(obs), info

	def step(self, action):
		action = torch.from_numpy(action)
		obs, reward, terminated, truncated, info = self.env.step(action)
		self._cumulative_reward += reward
		done = terminated or truncated
		info['terminated'] = terminated[0].detach().clone().cpu()
		return_value = (self._squeeze_obs(obs), reward[0].detach().clone().cpu(), terminated[0].detach().clone().cpu(), truncated[0].detach().clone().cpu(), info)
		del obs, action, terminated, truncated, info
		return return_value

	def _squeeze_obs(self, obs):
		new_obs = obs["policy"].detach().clone().cpu()#.squeeze()
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
		raise ValueError('Unknown task:', cfg.task)
	env = gym.make(ISAACLAB_TASKS[cfg.task], cfg=env_cfg)
	env = IsaacLabWrapper(env, cfg)
	env = gym.wrappers.FlattenObservation(env)
	env = FlattenAction(env)
	env = Timeout(env, max_episode_steps=1000)
	return env
