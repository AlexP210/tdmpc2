from copy import deepcopy

import torch
import torch.nn as nn

from tdmpc2.common import layers, math, init
from tensordict import TensorDict
from tensordict.nn import TensorDictParams


class WorldModel(nn.Module):
	"""
	TD-MPC2 implicit world model architecture.
	Can be used for both single-task and multi-task experiments.
	"""

	def __init__(self, cfg):
		super().__init__()
		self.cfg = cfg
		if cfg.multitask:
			self._task_emb = nn.Embedding(len(cfg.tasks), cfg.task_dim, max_norm=1)
			self.register_buffer("_action_masks", torch.zeros(len(cfg.tasks), cfg.action_dim))
			for i in range(len(cfg.tasks)):
				self._action_masks[i, :cfg.action_dims[i]] = 1.
		# 'dino_cap' swaps the whole per-stream CNN/MLP encoder and the fusion layer below for a
		# frozen DINOv3 backbone with a trainable CAP head, fed the same `rgb_state` observation.
		# Built after `weight_init` has run (see below) so that init cannot touch DINO's weights.
		self._dino_cap = cfg.get('encoder', 'cnn') == 'dino_cap'
		if self._dino_cap:
			assert cfg.obs == 'rgb_state' and not cfg.multitask, \
				"encoder='dino_cap' needs obs='rgb_state' and a single task"
			self._encoder = nn.Identity()
		else:
			self._encoder = layers.enc(cfg)
		# `obs_shape` has one key per observation stream and `layers.enc` builds an encoder for
		# each. With more than one -- e.g. obs='rgb_state', a camera plus a proprio vector -- their
		# latents are concatenated and projected back down to latent_dim, so everything downstream
		# (dynamics, reward, policy prior, Q) still sees a single latent_dim vector and needs no
		# change. Sorted so the concatenation order does not depend on dict iteration order, which
		# would otherwise silently permute a reloaded checkpoint's input.
		self._obs_keys = sorted(cfg.obs_shape.keys())
		self._fuse = None
		if len(self._obs_keys) > 1 and not self._dino_cap:
			assert not cfg.multitask, \
				'multi-stream observations and multitask task embeddings are not supported together'
			self._fuse = nn.Sequential(
				nn.Linear(sum(self._stream_dim(cfg, k) for k in self._obs_keys), cfg.latent_dim),
				layers.SimNorm(cfg),
			)
		self._dynamics = layers.mlp(cfg.latent_dim + cfg.action_dim + cfg.task_dim, 2*[cfg.mlp_dim], cfg.latent_dim, act=layers.SimNorm(cfg))
		self._reward = layers.mlp(cfg.latent_dim + cfg.action_dim + cfg.task_dim, 2*[cfg.mlp_dim], max(cfg.num_bins, 1))
		self._termination = layers.mlp(cfg.latent_dim + cfg.task_dim, 2*[cfg.mlp_dim], 1) if cfg.episodic else None
		self._pi = layers.mlp(cfg.latent_dim + cfg.task_dim, 2*[cfg.mlp_dim], 2*cfg.action_dim)
		self._Qs = layers.Ensemble([layers.mlp(cfg.latent_dim + cfg.action_dim + cfg.task_dim, 2*[cfg.mlp_dim], max(cfg.num_bins, 1), dropout=cfg.dropout) for _ in range(cfg.num_q)])
		self.apply(init.weight_init)
		init.zero_([self._reward[-1].weight, self._Qs.params["2", "weight"]])
		if self._dino_cap:
			from tdmpc2.common.dino_cap import DinoCAPEncoder
			self._encoder = DinoCAPEncoder(cfg)

		self.register_buffer("log_std_min", torch.tensor(cfg.log_std_min))
		self.register_buffer("log_std_dif", torch.tensor(cfg.log_std_max) - self.log_std_min)
		self.init()

	def _stream_dim(self, cfg, key):
		"""Output width of one stream's encoder, measured with a dry forward rather than derived.

		`layers.conv` has no final linear layer, so an rgb encoder's width is a function of the
		input resolution and `num_channels` through four conv layers; measuring it keeps this
		correct if either changes, where re-deriving the arithmetic here would not.
		"""
		shape = tuple(cfg.obs_shape[key])
		# the state encoder's mlp takes the task embedding concatenated on; the conv one does not
		x = torch.zeros(1, *shape) if len(shape) > 1 else torch.zeros(1, shape[0] + cfg.task_dim)
		with torch.no_grad():
			return self._encoder[key](x).shape[-1]

	def init(self):
		# Create params
		self._detach_Qs_params = TensorDictParams(self._Qs.params.data, no_convert=True)
		self._target_Qs_params = TensorDictParams(self._Qs.params.data.clone(), no_convert=True)

		# Create modules
		with self._detach_Qs_params.data.to("meta").to_module(self._Qs.module):
			self._detach_Qs = deepcopy(self._Qs)
			self._target_Qs = deepcopy(self._Qs)

		# Assign params to modules
		# We do this strange assignment to avoid having duplicated tensors in the state-dict -- working on a better API for this
		delattr(self._detach_Qs, "params")
		self._detach_Qs.__dict__["params"] = self._detach_Qs_params
		delattr(self._target_Qs, "params")
		self._target_Qs.__dict__["params"] = self._target_Qs_params

	def __repr__(self):
		repr = 'TD-MPC2 World Model\n'
		modules = ['Encoder', 'Fusion', 'Dynamics', 'Reward', 'Termination', 'Policy prior', 'Q-functions']
		for i, m in enumerate([self._encoder, self._fuse, self._dynamics, self._reward, self._termination, self._pi, self._Qs]):
			# covers both the single-stream case (no fusion) and non-episodic (no termination head)
			if m is None:
				continue
			repr += f"{modules[i]}: {m}\n"
		repr += "Learnable parameters: {:,}".format(self.total_params)
		return repr

	@property
	def total_params(self):
		return sum(p.numel() for p in self.parameters() if p.requires_grad)

	def to(self, *args, **kwargs):
		super().to(*args, **kwargs)
		self.init()
		return self

	def train(self, mode=True):
		"""
		Overriding `train` method to keep target Q-networks in eval mode.
		"""
		super().train(mode)
		self._target_Qs.train(False)
		return self

	def soft_update_target_Q(self):
		"""
		Soft-update target Q-networks using Polyak averaging.
		"""
		self._target_Qs_params.lerp_(self._detach_Qs_params, self.cfg.tau)

	def task_emb(self, x, task):
		"""
		Continuous task embedding for multi-task experiments.
		Retrieves the task embedding for a given task ID `task`
		and concatenates it to the input `x`.
		"""
		if isinstance(task, int):
			task = torch.tensor([task], device=x.device)
		emb = self._task_emb(task.long())
		if x.ndim == 3:
			emb = emb.unsqueeze(0).repeat(x.shape[0], 1, 1)
		elif emb.shape[0] == 1:
			emb = emb.repeat(x.shape[0], 1)
		return torch.cat([x, emb], dim=-1)

	def _encode_stream(self, key, x):
		"""One stream through its own encoder.

		The rgb branch loops over a leading time axis because `layers.conv` takes 4D input, while
		the mlp encoders broadcast over any number of leading dims.
		"""
		if key == 'rgb' and x.ndim == 5:
			return torch.stack([self._encoder[key](o) for o in x])
		return self._encoder[key](x)

	def encode(self, obs, task):
		"""
		Encodes an observation into its latent representation.
		A single-stream observation is a tensor. A multi-stream one is a dict (or TensorDict) keyed
		by `cfg.obs_shape`'s keys, whose per-stream latents are fused into one latent_dim vector.
		"""
		if self.cfg.multitask:
			obs = self.task_emb(obs, task)
		if self._dino_cap:
			return self._encoder(obs)
		if self._fuse is None:
			return self._encode_stream(self.cfg.obs, obs)
		return self._fuse(torch.cat(
			[self._encode_stream(k, obs[k]) for k in self._obs_keys], dim=-1))

	def next(self, z, a, task):
		"""
		Predicts the next latent state given the current latent state and action.
		"""
		if self.cfg.multitask:
			z = self.task_emb(z, task)
		z = torch.cat([z, a], dim=-1)
		return self._dynamics(z)

	def reward(self, z, a, task):
		"""
		Predicts instantaneous (single-step) reward.
		"""
		if self.cfg.multitask:
			z = self.task_emb(z, task)
		z = torch.cat([z, a], dim=-1)
		return self._reward(z)
	
	def termination(self, z, task, unnormalized=False):
		"""
		Predicts termination signal.
		"""
		assert task is None
		if self.cfg.multitask:
			z = self.task_emb(z, task)
		if unnormalized:
			return self._termination(z)
		return torch.sigmoid(self._termination(z))
		

	def pi(self, z, task):
		"""
		Samples an action from the policy prior.
		The policy prior is a Gaussian distribution with
		mean and (log) std predicted by a neural network.
		"""
		if self.cfg.multitask:
			z = self.task_emb(z, task)

		# Gaussian policy prior
		mean, log_std = self._pi(z).chunk(2, dim=-1)
		log_std = math.log_std(log_std, self.log_std_min, self.log_std_dif)
		eps = torch.randn_like(mean)

		if self.cfg.multitask: # Mask out unused action dimensions
			mean = mean * self._action_masks[task]
			log_std = log_std * self._action_masks[task]
			eps = eps * self._action_masks[task]
			action_dims = self._action_masks.sum(-1)[task].unsqueeze(-1)
		else: # No masking
			action_dims = None

		log_prob = math.gaussian_logprob(eps, log_std)

		# Scale log probability by action dimensions
		size = eps.shape[-1] if action_dims is None else action_dims
		scaled_log_prob = log_prob * size

		# Reparameterization trick
		action = mean + eps * log_std.exp()
		mean, action, log_prob = math.squash(mean, action, log_prob)

		entropy_scale = scaled_log_prob / (log_prob + 1e-8)
		info = TensorDict({
			"mean": mean,
			"log_std": log_std,
			"action_prob": 1.,
			"entropy": -log_prob,
			"scaled_entropy": -log_prob * entropy_scale,
		})
		return action, info

	def Q(self, z, a, task, return_type='min', target=False, detach=False):
		"""
		Predict state-action value.
		`return_type` can be one of [`min`, `avg`, `all`]:
			- `min`: return the minimum of two randomly subsampled Q-values.
			- `avg`: return the average of two randomly subsampled Q-values.
			- `all`: return all Q-values.
		`target` specifies whether to use the target Q-networks or not.
		"""
		assert return_type in {'min', 'avg', 'all'}

		if self.cfg.multitask:
			z = self.task_emb(z, task)

		z = torch.cat([z, a], dim=-1)
		if target:
			qnet = self._target_Qs
		elif detach:
			qnet = self._detach_Qs
		else:
			qnet = self._Qs
		out = qnet(z)

		if return_type == 'all':
			return out

		qidx = torch.randperm(self.cfg.num_q, device=out.device)[:2]
		Q = math.two_hot_inv(out[qidx], self.cfg)
		if return_type == "min":
			return Q.min(0).values
		return Q.sum(0) / 2
