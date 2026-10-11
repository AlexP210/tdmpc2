import types
from collections import deque

import gymnasium as gym
import numpy as np
import torch

from tdmpc2.common.dino_cap import build_vision_encoder


class DinoFeatures(gym.Wrapper):
	"""Replace the frame-stacked `rgb` observation with its frozen-DINOv3 patch features.

	Sits after `Pixels`, whose `rgb` is `(num_frames*3, H, W)` uint8, and emits
	`{'dino_patch_features': (num_frames, num_tokens, token_dim) fp16, 'state': ...}` for
	`DinoCAPEncoder` with `cfg.dino_cache`. DINO runs on the *newest* frame only, once per env
	step, and the last `num_frames` results are stacked the way `Pixels` stacks the frames -- so
	the planner and the replay buffer share one DINO forward per step, and the update step runs
	none. After a reset the stack holds copies of the first frame's features, as `Pixels` does.

	The backbone is the one `DinoCAPEncoder` builds (`build_vision_encoder`), so a cached feature
	is exactly what the uncached encoder would have computed from the same frame, up to fp16
	rounding. fp16 keeps a stacked observation the size of the uint8 rgb stack it replaces.
	"""

	def __init__(self, env, cfg, num_frames=3):
		super().__init__(env)
		self.cfg = cfg
		self.env = env
		space = env.observation_space
		channels, height, width = space['rgb'].shape
		assert channels == num_frames * 3, (channels, num_frames)
		self._device = torch.device(cfg.device)
		self._encoder = build_vision_encoder(cfg, types.SimpleNamespace(
			observation_dimension={'rgb': (1, 3, height, width)}))
		# built in train mode, where DINOv3 jitters its RoPE and so encodes a frame differently
		# on every call
		self._encoder.eval()
		self._features = deque([], maxlen=num_frames)
		self.observation_space = gym.spaces.Dict({
			'dino_patch_features': gym.spaces.Box(
				low=-np.inf, high=np.inf,
				shape=(num_frames, self._encoder.num_tokens, self._encoder.token_length),
				dtype=np.float16),
			'state': space['state'],
		})

	@torch.no_grad()
	def _frame_features(self, frame):
		"""`(3, H, W)` uint8 -> `(num_tokens, token_dim)` fp16 on the CPU."""
		out = self._encoder.encode({'rgb': frame.to(self._device).unsqueeze(0)})
		return out[0].half().cpu()  # (1, num_tokens, token_dim) -> (num_tokens, token_dim)

	def _get_obs(self, obs, is_reset=False):
		features = self._frame_features(obs['rgb'][-3:])
		for _ in range(self._features.maxlen if is_reset else 1):
			self._features.append(features)
		return {'dino_patch_features': torch.stack(list(self._features)), 'state': obs['state']}

	def reset(self):
		return self._get_obs(self.env.reset(), is_reset=True)

	def step(self, action):
		obs, reward, done, info = self.env.step(action)
		return self._get_obs(obs), reward, done, info
