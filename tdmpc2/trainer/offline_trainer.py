import os
from copy import deepcopy
from time import time
from pathlib import Path
from glob import glob

import numpy as np
import torch
from tqdm import tqdm

from common.buffer import Buffer
from trainer.base import Trainer

import torch.nn.functional as F
from tdmpc2.envs.tasks import cheetah, walker, hopper, reacher, ball_in_cup, pendulum, fish
from dm_control import suite
suite.ALL_TASKS = suite.ALL_TASKS + suite._get_tasks('custom')
suite.TASKS_BY_DOMAIN = suite._get_tasks_by_domain(suite.ALL_TASKS)
from dm_control.suite.wrappers import action_scale

from tdmpc2.envs.wrappers.timeout import Timeout
from tdmpc2.envs.dmcontrol import DMControlWrapper
from tdmpc2.envs.wrappers.tensor import TensorWrapper


class OfflineTrainer(Trainer):
	"""Trainer class for multi-task offline TD-MPC2 training."""

	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self._start_time = time()
		self.best_avg_return_across_eval_tasks = -float("inf")
		self._step = 0
	
	def eval(self):
		"""Evaluate a TD-MPC2 agent."""
		results = dict()
		task_conversion = {
			"cheetah-flip": "cheetah-jump"
		}
		eval_task_idxs = []
		eval_task_envs = []
		for eval_task_name in self.cfg.eval_tasks:
			if eval_task_name in self.cfg.tasks:
				eval_task_idxs.append(self.cfg.tasks.index(eval_task_name))
			else:
				eval_task_idxs.append(self.cfg.tasks.index(task_conversion[eval_task_name]))
			eval_task_envs.append(self.make_eval_env(eval_task_name))
		for task_name, task_idx, task_env in tqdm(zip(self.cfg.eval_tasks, eval_task_idxs, eval_task_envs), desc='Evaluating'):
			ep_rewards, ep_successes = [], []

			model_obs_shape = self.cfg.obs_shape["state"]
			env_action_shape = task_env.action_space.shape

			for i in range(self.cfg.eval_episodes):
				obs, done, ep_reward, t = task_env.reset(task_idx), False, 0, 0
				obs = self.pad_dims(obs, model_obs_shape, len(model_obs_shape))
				if self.cfg.save_video:
					self.logger.video.init(task_env, enabled=(i==0))

				while not done:
					torch.compiler.cudagraph_mark_step_begin()
					action = self.agent.act(obs, t0=t==0, eval_mode=True, task=task_idx)
					action = self.unpad_dims(action, env_action_shape, len(env_action_shape))
					obs, reward, done, info = task_env.step(action)
					obs = self.pad_dims(obs, model_obs_shape, len(model_obs_shape))
					ep_reward += reward
					t += 1
					if self.cfg.save_video:
						self.logger.video.record(task_env)
				ep_rewards.append(ep_reward)
				ep_successes.append(info['success'])
				if self.cfg.save_video:
					self.logger.video.save(self._step)
			results.update({
				f'episode_reward+{task_name}': np.nanmean(ep_rewards),
				f'episode_success+{task_name}': np.nanmean(ep_successes),})
		return results
	
	def make_eval_env(self, task_name):
		domain, task = task_name.replace('-', '_').split('_', 1)
		domain = dict(cup='ball_in_cup', pointmass='point_mass').get(domain, domain)
		env = suite.load(domain,
						task,
						task_kwargs={'random': self.cfg.seed},
						visualize_reward=False)
		env = action_scale.Wrapper(env, minimum=-1., maximum=1.)
		env = DMControlWrapper(env, domain)
		env = Timeout(env, max_episode_steps=500)
		env = TensorWrapper(env)
		return env

	def _load_dataset(self):
		"""Load dataset for offline training."""
		fp = Path(os.path.join(self.cfg.data_dir, self.cfg.data_name))
		fps = sorted(glob(str(fp)))
		assert len(fps) > 0, f'No data found at {fp}'
		print(f'Found {len(fps)} files in {fp}')
		if len(fps) < (20 if self.cfg.task == 'mt80' else 4):
			print(f'WARNING: expected 20 files for mt80 task set, 4 files for mt30 task set, found {len(fps)} files.')
	
		# Create buffer for sampling
		_cfg = deepcopy(self.cfg)
		# _cfg.episode_length = 101 if self.cfg.task == 'mt80' else 501
		# _cfg.buffer_size = 550_450_000 if self.cfg.task == 'mt80' else 345_690_000
		# _cfg.steps = _cfg.buffer_size
		# self.buffer = Buffer(_cfg)

		# Do a pass through all files to find the episode length and number of samples for the dataset
		_cfg.episode_length = None
		_cfg.buffer_size = 0
		for fp in tqdm(fps, desc='Loading data'):
			td = torch.load(fp, weights_only=False)
			if _cfg.episode_length is None:
				_cfg.episode_length = td.shape[1]
			elif _cfg.episode_length != td.shape[1]: 
				raise ValueError(
					f"Data files have incongurous episode lengths: {fps}"
				)
			_cfg.buffer_size += td.shape[0] * td.shape[1]

		# Create the buffer to store the data
		_cfg.steps = _cfg.buffer_size
		self.buffer = Buffer(_cfg)

		# Load the data
		for fp in tqdm(fps, desc='Loading data'):
			td = torch.load(fp, weights_only=False)
			# To make it compatible with the data shapes used for TSD
			td["reward"] = td["reward"].squeeze()
			td["task"] = td["task"].squeeze()
			# If we are multi-task, then pad the observations/actions to the right shape
			obs_shape = self.cfg.obs_shape["state"]
			action_dim = self.cfg.action_dim
			td["obs"] = self.pad_dims(td["obs"], obs_shape, len(obs_shape))
			td["action"] = self.pad_dims(td["action"], (action_dim,), 1)

			self.buffer.load(td)
			
		expected_episodes = _cfg.buffer_size // _cfg.episode_length
		if self.buffer.num_eps != expected_episodes:
			print(f'WARNING: buffer has {self.buffer.num_eps} episodes, expected {expected_episodes} episodes for {self.cfg.task} task set.')

	def pad_dims(self, tensor, padded_dims, n_dims):
		# Current O shape
		shape = tensor.shape[-n_dims:]
		b_shape = tensor.shape[:-n_dims]
		
		# Compute padding needed per dim (pad_last, pad_second_last, ...)
		# torch.nn.functional.pad expects (last_dim_pad, second_last_dim_pad, ...)
		padding = []
		for size, target in zip(reversed(shape), reversed(padded_dims)):
			padding += [0, target - size]		
		return F.pad(tensor, padding)

	def unpad_dims(self, tensor, original_dims, n_dims):
		slices = [slice(None)] * (len(tensor.shape) - n_dims)  # keep B dims
		slices += [slice(0, s) for s in original_dims]          # slice O dims
		return tensor[tuple(slices)]

	def train(self):
		"""Train a TD-MPC2 agent."""
		# assert self.cfg.multitask and self.cfg.task in {'mt30', 'mt80'}, \
		# 	'Offline training only supports multitask training with mt30 or mt80 task sets.'
		self._load_dataset()
		
		print(f'Training agent for {self.cfg.steps} iterations...')
		metrics = {}
		for i in range(self.cfg.steps):
			self._step = i
			# Update agent
			train_metrics = self.agent.update(self.buffer)

			# Evaluate agent periodically
			if i % self.cfg.eval_freq == 0 or i % 10_000 == 0:
				metrics = {
					'iteration': i,
					'elapsed_time': time() - self._start_time,
				}
				metrics.update(train_metrics)
				if i % self.cfg.eval_freq == 0:
					metrics.update(self.eval())
					avg_return_across_eval_tasks = np.mean(
						[metrics[key] for key in metrics.keys() if "episode_reward+" in key]
					)
					self.logger.pprint_multitask(metrics, self.cfg)
					if i > 0 and avg_return_across_eval_tasks > self.best_avg_return_across_eval_tasks:
						self.logger.save_agent(self.agent, identifier='best')
						self.best_avg_return_across_eval_tasks = avg_return_across_eval_tasks
				self.logger.log(metrics, 'pretrain')
			
		self.logger.finish(self.agent)
