import os
import pathlib
import types

import torch
import torch.nn as nn
from omegaconf import OmegaConf

from tdmpc2.common import layers


def build_vision_encoder(cfg, task):
	"""S2P's frozen DINOv3 patch-token encoder, as `DinoCAPEncoder` and the feature-caching env
	wrapper (`envs/wrappers/dino_features.py`) both need it -- the cache is only valid if the two
	are the same function, so they build it in one place."""
	# `dinov3_passthrough_encoder_model` reads PROJECT_ROOT when it is first imported, to find
	# dependencies/dinov3, and TD-MPC2 runs do not export it. Fill it in from this file's
	# location (agents/tdmpc2/tdmpc2/common/ -> four levels up) before the import below, but
	# never override one that is already set.
	os.environ.setdefault('PROJECT_ROOT', str(pathlib.Path(__file__).resolve().parents[4]))
	from s2p.models.dinov3_passthrough_encoder_model import DINOV3PassthroughEncoderModel
	return DINOV3PassthroughEncoderModel(OmegaConf.create(dict(
		resize_size=cfg.dino_resize,
		model_name=cfg.dino_model,
		token_mode='patch',
		freeze=True,
		checkpoint=cfg.dino_checkpoint,
		device=cfg.device,
		observation_key='rgb',
	)), task)


class DinoCAPEncoder(nn.Module):
	"""
	A frozen DINOv3 backbone with a trainable cross-attention-pooling (CAP) head, built directly
	from S2P's classes so it is the same encoder S2P's `train_dino_deterministic` uses:

		TeacherStudentEncoderModel(
			teacher=JointEncoderModel([DINOv3 patch tokens of `rgb`, identity of `proprio`]),
			student=DeterministicCAPStudentEncoderModel,
		)

	followed by a `SimNorm`, since the rest of TD-MPC2 (dynamics, consistency loss) assumes a
	simplicial latent and the CAP head's own output is unconstrained.

	Takes the `obs='rgb_state'` dict the env already produces -- `rgb`, the camera frames
	channel-stacked as `(..., num_frames*3, H, W)` uint8, and `state`, the proprio vector -- and
	returns `(..., latent_dim)`. The frames are split back onto their own axis, which is the
	layout S2P's encoders take. The single proprio vector is broadcast over every frame and
	patch token by `JointEncoderModel`, so nothing about the env or the buffer changes.

	Only the CAP head trains. With `cfg.dino_cache` the env has already run DINO (see
	`envs/wrappers/dino_features.py`) and the observation carries `dino_patch_features` in place
	of `rgb`, which S2P's backbone returns as-is. Otherwise DINO re-encodes every frame of every
	call.
	"""

	def __init__(self, cfg):
		super().__init__()
		from s2p.models.identity_encoder import IdentityEncoder
		from s2p.models.joint_encoder_model import JointEncoderModel
		from s2p.models.deterministic_cap_student_encoder_model import DeterministicCAPStudentEncoderModel
		from s2p.models.teacher_student_encoder_model import TeacherStudentEncoderModel

		self.cached = bool(cfg.get('dino_cache', False))
		if self.cached:
			# `rgb` is gone from the observation; the teacher still needs an image shape to size
			# itself from a dummy pass, which is the camera's
			self.num_frames, num_tokens, token_dim = cfg.obs_shape['dino_patch_features']
			height = width = cfg.camera_resolution
		else:
			channels, height, width = cfg.obs_shape['rgb']
			assert channels % 3 == 0, f'expected channel-stacked RGB frames, got {channels} channels'
			self.num_frames = channels // 3
		# S2P's encoders only read `task.observation_dimension`, to size themselves from a dummy
		# forward pass, so this stands in for the task object they were written against.
		task = types.SimpleNamespace(observation_dimension={
			'rgb': (self.num_frames, 3, height, width),
			'proprio': tuple(cfg.obs_shape['state']),
		})

		vision = build_vision_encoder(cfg, task)
		proprio = IdentityEncoder(OmegaConf.create(dict(observation_key='proprio')), task)
		teacher = JointEncoderModel(OmegaConf.create(dict(
			downprojection_dim=cfg.latent_dim,
			project=False,
			freeze=False,
			checkpoint=None,
			device=cfg.device,
		)), [vision, proprio], task)
		student = DeterministicCAPStudentEncoderModel(OmegaConf.create(dict(
			embedding_dim=cfg.cap_embedding_dim,
			n_heads=cfg.cap_n_heads,
			latent_dim=cfg.latent_dim,
			n_queries=cfg.cap_n_queries,
			ff_hidden_layers=cfg.cap_ff_hidden_layers,
			ff_hidden_dim=cfg.cap_ff_hidden_dim,
			n_blocks=cfg.cap_n_blocks,
			ema_decay=0.99,  # S2P's EMA target encoder, which TD-MPC2 does not use
			device=cfg.device,
			checkpoint=None,
			freeze=False,
		)), task, teacher)
		self.encoder = TeacherStudentEncoderModel(OmegaConf.create(dict(
			freeze=False,
			checkpoint=None,
			device=cfg.device,
		)), student, teacher)

		if self.cached:
			expected = (self.num_frames, num_tokens, token_dim + cfg.obs_shape['state'][0])
			assert tuple(student.teacher_output_dim[-3:]) == expected, (
				f'the cached features {(self.num_frames, num_tokens, token_dim)} do not match what '
				f'the backbone produces: teacher output is {tuple(student.teacher_output_dim)}')

		# The whole teacher is frozen, not just the backbone: the joint encoder also builds an
		# unused projection layer, which would otherwise count as a learnable parameter.
		for p in teacher.parameters():
			p.requires_grad_(False)

		self.simnorm = layers.SimNorm(cfg)

	def forward(self, obs):
		if self.cached:
			# stored fp16 in the buffer, float for the head
			frames = {'dino_patch_features': obs['dino_patch_features'].float()}
		else:
			rgb = obs['rgb']
			# the backbone's transform scales uint8 to [0, 1]; on float input that is a no-op and
			# [0, 255] would be ImageNet-normalized into a different feature space
			assert rgb.dtype == torch.uint8, f'expected uint8 frames, got {rgb.dtype}'
			frames = {'rgb': rgb.unflatten(-3, (self.num_frames, 3))}
		z = self.encoder.encode({**frames, 'proprio': obs['state'].float()})
		return self.simnorm(z)
