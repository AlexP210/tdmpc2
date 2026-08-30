"""TD-MPC2 on this project's ManiSkill tasks.

The env itself -- task ids, camera views, backend/device pinning, termination handling -- comes from
`environments/custom_maniskill_tasks`, the same builder TSD's online rollouts and every recorded
dataset here use, so a TD-MPC2 baseline runs against the same env as the methods it is compared
against rather than a second copy of the setup that has drifted from it.

What stays in this file is the TD-MPC2 side of the interface: ManiSkill's batched-by-`num_envs`
torch observations and gymnasium's five-tuple reduced to the unbatched four-tuple TD-MPC2's
trainer, buffer and `TensorWrapper` expect.

Two of the builder's wrappers are deliberately left off:

- `FrameSkip`, because it makes the action space the *concatenation* of `frame_skip` primitive
  actions (the open-loop chunk DINO-WM plans in). TD-MPC2 plans one primitive action per step, so
  repeating an action -- what `action_repeat` does below -- is the equivalent knob here.
- `FrameStack`, because it stacks along a new frame axis, while TD-MPC2's conv encoder wants frames
  concatenated into the channel axis. The `Pixels` wrapper below is that stack, mirroring
  `dmcontrol.py`, with one difference noted in its docstring: frames are the observation camera's,
  not `render()`'s.

`IgnoreTerminations` *is* used, via `make_env(ignore_terminations=True)`: ManiSkill recomputes
`terminated` from the success predicate every step, so honouring it would end an episode the first
time the goal is touched. It is redundant for the `-v1.1` ids (whose tasks already never terminate)
and load-bearing for the stock ones.
"""

import math
from collections import deque

import gymnasium as gym
import numpy as np
import torch

from mani_skill.utils.registration import REGISTERED_ENVS

# The one definition of these tasks -- see the module docstring. Imported after `dmcontrol` in
# `envs/__init__.py` on purpose: this pulls in torch/sapien, and dm_control has to initialize EGL
# before torchrl-style imports do or headless rendering fails.
from custom_maniskill_tasks import (
	CONTROL_MODE,
	FOCUSED_CAMERA_UID,
	WRIST_CAMERA_UID,
	make_env as make_custom_maniskill_env,
)

from tdmpc2.envs.wrappers.timeout import Timeout


CUSTOM_MANISKILL_TASKS = {
	# The `-v1.1` ids are this project's own (registered by importing custom_maniskill_tasks): the
	# stock tasks with early termination removed, which is the convention every offline dataset
	# here was collected under.
	'push-cube': dict(env='PushCube-v1.1', control_mode='pd_ee_delta_pos'),
	'place-sphere': dict(env='PlaceSphere-v1.1', control_mode='pd_ee_delta_pos'),
	'lift-peg-upright': dict(env='LiftPegUpright-v1.1', control_mode='pd_ee_delta_pos'),
}


class CustomManiSkillWrapper(gym.Wrapper):
	"""ManiSkill's API as the unbatched four-tuple TD-MPC2 expects.

	Four things change:

	- `num_envs=1`'s leading axis comes off every observation, action, reward and flag, since
	  TD-MPC2's trainer, buffer and planner are all single-env.
	- gymnasium's `(obs, reward, terminated, truncated, info)` becomes `(obs, reward, done, info)`,
	  with `terminated` and `success` moved into `info` as floats -- the two keys `OnlineTrainer`
	  and `TensorWrapper` read. `terminated` is always 0 here (see the module docstring), so `done`
	  is truncation alone, which is why `episodic=false` is the right setting for these tasks.
	- under `obs='rgb'` the observation is the camera frame `camera_view` selects, as a CHW uint8
	  tensor; `Pixels` stacks it. Under `obs='state'` it is ManiSkill's flattened state vector.
	- the action space is symmetrized to one box over all dimensions, as `maniskill.py` does,
	  because TD-MPC2's policy is a tanh over a single shared range.

	`action_repeat` applies each action that many consecutive sim steps, summing reward, the way
	`dmcontrol.py` and `maniskill.py` both do. It defaults to 1: these tasks' time limits are
	already short (50 primitive steps for PushCube) and the recorded datasets are collected at the
	primitive rate.
	"""

	def __init__(self, env, cfg, action_repeat=1):
		super().__init__(env)
		self.env = env
		self.cfg = cfg
		# an assert rather than a ValueError: `envs/__init__.py` swallows ValueError as "not my
		# task" and would report a misconfigured action_repeat as a missing environment
		assert action_repeat >= 1, f'action_repeat must be >= 1, got {action_repeat}'
		self.action_repeat = action_repeat
		self._seed = cfg.get('seed', None)

		base = env.unwrapped
		self._camera_uid = self._resolve_camera_uid(env) if cfg.obs == 'rgb' else None
		if self._camera_uid is not None:
			height, width, channels = base.single_observation_space[
				'sensor_data'][self._camera_uid]['rgb'].shape
			self.observation_space = gym.spaces.Box(
				low=0, high=255, shape=(channels, height, width), dtype=np.uint8)
		else:
			self.observation_space = base.single_observation_space

		action_space = base.single_action_space
		self.action_space = gym.spaces.Box(
			low=np.full(action_space.shape, action_space.low.min()),
			high=np.full(action_space.shape, action_space.high.max()),
			dtype=action_space.dtype,
		)

	def _resolve_camera_uid(self, env):
		"""Which camera in the built scene the observations come from.

		Normally there is exactly one: `camera_view='wrist'` drops the task's own camera by default
		and the other views add none. The fallback is only reached when a task registers several
		external cameras, or when `wrist_only=False` kept both -- and then guessing is worse than
		failing, since a policy trained on the wrong view is a silent failure.
		"""
		sensors = env.unwrapped.single_observation_space['sensor_data']
		uids = [uid for uid, space in sensors.spaces.items() if 'rgb' in space.spaces]
		if len(uids) == 1:
			return uids[0]
		preferred = WRIST_CAMERA_UID if self.cfg.get('camera_view') == 'wrist' else FOCUSED_CAMERA_UID
		if preferred in uids:
			return preferred
		raise ValueError(
			f'Cannot tell which camera to observe {self.cfg.task} through: it renders {uids}, and '
			f'none of them is the {preferred!r} that camera_view='
			f'{self.cfg.get("camera_view", "default")!r} configures.'
		)

	def _observation(self, obs):
		if self._camera_uid is not None:
			rgb = obs['sensor_data'][self._camera_uid]['rgb']
			# (1, H, W, C) -> (C, H, W). The permute is never contiguous, so `contiguous()` is a
			# real copy -- required rather than defensive: ManiSkill hands back its camera capture
			# buffer itself and overwrites it in place on the next step, while TD-MPC2 keeps a
			# whole episode of observations alive in `_tds`.
			return rgb[0].permute(2, 0, 1).contiguous().cpu()
		# ManiSkill allocates the flattened state fresh per step, but copy for the same reason:
		# nothing here should depend on when the sim happens to reuse a buffer.
		return obs[0].float().clone().cpu()

	def _info(self, info):
		success = info.get('success', 0.0)
		return dict(
			success=float(_scalar(success)),
			# always 0.0: terminations are suppressed underneath this wrapper
			terminated=0.0,
		)

	def reset(self):
		"""Reset, seeding only the first episode.

		ManiSkill takes no seed at construction: `reset(seed=...)` is what seeds the episode RNG,
		and re-passing the same seed every reset would make every episode identical. Seeding once
		and letting the stream run is what `dmcontrol.py` gets from `task_kwargs={'random': seed}`.
		"""
		kwargs = {} if self._seed is None else dict(seed=int(self._seed))
		self._seed = None
		obs, _ = self.env.reset(**kwargs)
		return self._observation(obs)

	def step(self, action):
		action = torch.as_tensor(np.asarray(action), dtype=torch.float32).reshape(1, -1)
		reward = 0.0
		for _ in range(self.action_repeat):
			obs, r, terminated, truncated, info = self.env.step(action)
			reward += float(_scalar(r))
			done = bool(_scalar(terminated)) or bool(_scalar(truncated))
			if done:
				# stop rather than run past the time limit, as FrameSkip does for the same reason
				break
		return self._observation(obs), reward, done, self._info(info)

	def render(self, *args, **kwargs):
		"""An HWC uint8 frame for `logger.video`.

		From ManiSkill's separate human render camera, so it is unaffected by `camera_view` and by
		the observation resolution: the video shows the scene, not the policy's input. Its size is
		fixed by the task's render camera config, hence the ignored width/height arguments that
		`dmcontrol.py`'s renderer accepts.
		"""
		return np.asarray(self.env.render()[0].cpu())

	def __getattr__(self, name):
		"""
		If this env does not have the attribute, then we try to
		recursively access that attribute from inner envs.

		gymnasium >= 1.0 dropped `Wrapper.__getattr__`, so without this a caller reaching through
		for ManiSkill's own API (`get_state_dict`, `control_freq`, ...) sees only this wrapper.
		"""
		if name.startswith('_'):
			raise AttributeError(name)
		env = self.env
		while not hasattr(env, name):
			if hasattr(env, 'env'): # while the env is still wrapped,
				env = env.env
			else: # reached the innermost env and still didn't find it.
				raise AttributeError(f'{env} has no attribute {name}.')
		return getattr(env, name) # reached if env **has** attribute name.


class Pixels(gym.Wrapper):
	"""Frame-stack the camera observation into TD-MPC2's `(num_frames*3, size, size)` uint8 obs.

	The same channel-concatenated stack as `dmcontrol.py`'s `Pixels` -- including holding
	`num_frames` copies of the first frame after a reset -- and defined here rather than imported
	from it so that a ManiSkill run does not drag `dm_control` (and its EGL initialization) into
	the process.

	It differs in where a frame comes from. `dmcontrol.py` renders one at the size it wants;
	ManiSkill's camera resolution is fixed when the scene is built, and `render()` uses the human
	render camera, which is *not* the camera `camera_view` selects. So this stacks the observation
	the env already returns, leaving `make_env(camera_resolution=...)` in charge of the size --
	which is what keeps a TD-MPC2 run's input the same view TSD and the recorded datasets use.
	Note TD-MPC2's conv encoder asserts 64x64 (`common/layers.py`), so that resolution is not free
	to change without changing the encoder.
	"""

	def __init__(self, env, cfg, num_frames=3):
		super().__init__(env)
		self.cfg = cfg
		self.env = env
		channels, height, width = env.observation_space.shape
		self.observation_space = gym.spaces.Box(
			low=0, high=255, shape=(num_frames*channels, height, width), dtype=np.uint8)
		self._frames = deque([], maxlen=num_frames)

	def _get_obs(self, frame, is_reset=False):
		for _ in range(self._frames.maxlen if is_reset else 1):
			self._frames.append(frame)
		return torch.cat(list(self._frames))

	def reset(self):
		return self._get_obs(self.env.reset(), is_reset=True)

	def step(self, action):
		frame, reward, done, info = self.env.step(action)
		return self._get_obs(frame), reward, done, info

	def __getattr__(self, name):
		"""
		If this env does not have the attribute, then we try to
		recursively access that attribute from inner envs.
		"""
		if name.startswith('_'):
			raise AttributeError(name)
		env = self.env
		while not hasattr(env, name):
			if hasattr(env, 'env'): # while the env is still wrapped,
				env = env.env
			else: # reached the innermost env and still didn't find it.
				raise AttributeError(f'{env} has no attribute {name}.')
		return getattr(env, name) # reached if env **has** attribute name.


def _scalar(x):
	"""A python number from ManiSkill's per-env `(num_envs,)` tensors."""
	if isinstance(x, torch.Tensor):
		return x.reshape(-1)[0].item()
	if isinstance(x, np.ndarray):
		return x.reshape(-1)[0].item()
	return x


def make_env(cfg):
	"""
	Make a ManiSkill environment from environments/custom_maniskill_tasks.
	"""
	if cfg.task in CUSTOM_MANISKILL_TASKS:
		task_cfg = CUSTOM_MANISKILL_TASKS[cfg.task]
	elif cfg.task in REGISTERED_ENVS:
		# any other registered ManiSkill id, spelled out in full, at the project's control mode
		task_cfg = dict(env=cfg.task, control_mode=CONTROL_MODE)
	else:
		raise ValueError('Unknown task:', cfg.task)
	assert cfg.obs in {'state', 'rgb'}, 'This task only supports state and rgb observations.'

	action_repeat = 1#int(cfg.get('action_repeat', 1))
	env_kwargs = dict(
		obs_mode=cfg.obs,
		control_mode=task_cfg["control_mode"],
		num_envs=1,
		camera_view="wrist",
		# 64 rather than the builder's 224 default: TD-MPC2's conv encoder asserts 64x64
		camera_resolution=64,
		# TD-MPC2 acts one primitive action at a time and stacks in the channel axis, so neither
		# of the builder's frame wrappers is applied -- see the module docstring.
		frame_skip=1,
		n_frames=None,
		ignore_terminations=True,
		render_mode='rgb_array',
		sim_backend=f"physx_cuda:{cfg.device.split(':')[1]}",
	)

	# TEMPORARY (2026-08-26): swap the task's dense reward for DINO-feature progress towards a
	# per-episode goal image. Delete this block and `envs/dino_goal_reward.py` to revert.
	reward_wrappers = ()
	if cfg.dino_reward:
		assert cfg.task == 'push-cube', \
			f'dino_reward only builds a goal image for push-cube, not {cfg.task}'
		from tdmpc2.envs.dino_goal_reward import reward_wrappers as dino_reward_wrappers
		reward_wrappers = dino_reward_wrappers(
			cfg,
			# the mirror env must match the training env exactly, or the goal image would be
			# rendered through a different camera than the observations
			env_factory=lambda: make_custom_maniskill_env(task_cfg['env'], **env_kwargs),
			camera_uid=WRIST_CAMERA_UID,
		)

	env = make_custom_maniskill_env(
		task_cfg['env'], reward_wrappers=reward_wrappers, **env_kwargs
	)
	env = CustomManiSkillWrapper(env, cfg, action_repeat=action_repeat)
	if cfg.obs == 'rgb':
		env = Pixels(env, cfg, num_frames=int(cfg.get('num_frames', 3)))
	# The task's own registered limit rather than a hardcoded one: 50 steps for PushCube and
	# friends, 100 for PegInsertionSide-v1. In macro steps, and rounded up because the episode does
	# reach a final partial chunk when action_repeat does not divide the limit.
	primitive_steps = REGISTERED_ENVS[task_cfg['env']].max_episode_steps
	return Timeout(env, max_episode_steps=math.ceil(primitive_steps / action_repeat))
