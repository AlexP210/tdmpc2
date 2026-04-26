import gymnasium as gym


class Timeout(gym.Wrapper):
	"""
	Wrapper for enforcing a time limit on the environment.
	"""

	def __init__(self, env, max_episode_steps):
		super().__init__(env)
		self._max_episode_steps = max_episode_steps
	
	@property
	def max_episode_steps(self):
		return self._max_episode_steps

	def reset(self, **kwargs):
		self._t = 0
		return self.env.reset(**kwargs)

	def step(self, action):
		obs, reward, done, info = self.env.step(action)
		self._t += 1
		done = done or self._t >= self.max_episode_steps
		return obs, reward, done, info

	def __getattr__(self, name):
		"""
		If this env does not have the attribute, then we try to 
		recursively access that attribute from inner envs.
		"""
		env = self.env
		while not hasattr(env, name):
			if hasattr(env, 'env'): # while the env is still wrapped,
				env = env.env
			else: # reached the innermost env and still didn't find it.
				raise AttributeError(f'{env} has no attribute {name}.')
		return getattr(env, name) # reached if env **has** attribute name.