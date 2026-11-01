import random
from time import time

import numpy as np
import torch
from tensordict.tensordict import TensorDict
from termcolor import colored

from tdmpc2.common import resume
from trainer.base import Trainer

CHECKPOINT_FREQ = 10_000  # gradient updates between periodic checkpoints


class OnlineTrainer(Trainer):
	"""Trainer class for single-task online TD-MPC2 training."""

	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self._step = 0
		self._ep_idx = 0
		self._grad_step = 0
		self._start_time = time()
		# the backup's buffer chunks, oldest first (see `save_resume_state`)
		self._buffer_chunks = []
		self._resumed = False

	def common_metrics(self):
		"""Return a dictionary of current metrics."""
		elapsed_time = time() - self._start_time
		return dict(
			step=self._step,
			episode=self._ep_idx,
			elapsed_time=elapsed_time,
			steps_per_second=self._step / elapsed_time
		)

	def eval(self):
		"""Evaluate a TD-MPC2 agent."""
		ep_rewards, ep_successes, ep_lengths = [], [], []
		for i in range(self.cfg.eval_episodes):
			obs, done, ep_reward, t = self.env.reset(), False, 0, 0
			if self.cfg.save_video:
				self.logger.video.init(self.env, enabled=(i==0))
			while not done:
				torch.compiler.cudagraph_mark_step_begin()
				action = self.agent.act(obs, t0=t==0, eval_mode=True)
				obs, reward, done, info = self.env.step(action)
				ep_reward += reward
				t += 1
				if self.cfg.save_video:
					self.logger.video.record(self.env)
			ep_rewards.append(ep_reward)
			ep_successes.append(info['success'])
			ep_lengths.append(t)
			if self.cfg.save_video:
				self.logger.video.save(self._step)
		return dict(
			episode_reward=np.nanmean(ep_rewards),
			episode_success=np.nanmean(ep_successes),
			episode_length= np.nanmean(ep_lengths),
		)

	def save_resume_state(self):
		"""Back up everything `resume` needs to continue from this episode boundary.

		Called right after an eval, once the finished episode is in the buffer. Only the transitions
		added since the previous backup are written; older chunks are dropped once newer ones cover
		the buffer's capacity, since by then the storage has overwritten them.
		"""
		start = time()
		chunk = self.buffer.save_new(resume.buffer_directory(self.cfg))
		if chunk is not None:
			self._buffer_chunks.append(chunk)
		kept, covered = [], 0
		for chunk in reversed(self._buffer_chunks):
			if covered >= self.buffer.capacity:
				break
			kept.insert(0, chunk)
			covered += chunk['length']
		self._buffer_chunks = kept
		resume.save(self.cfg, dict(
			step=self._step,
			ep_idx=self._ep_idx,
			grad_step=self._grad_step,
			elapsed_time=time() - self._start_time,
			agent=self.agent.training_state(),
			buffer_chunks=self._buffer_chunks,
			num_eps=self.buffer.num_eps,
			logger=self.logger.state_dict(),
			rng=dict(
				python=random.getstate(),
				numpy=np.random.get_state(),
				torch=torch.get_rng_state(),
				cuda=torch.cuda.get_rng_state(self.cfg.device),
			),
		), self._buffer_chunks)
		print(colored(f'Backed up for resuming at step {self._step:,} in {time() - start:.1f}s.', 'yellow'))

	def resume(self, state):
		"""Continue from a backup `save_resume_state` wrote (the Logger is resumed separately)."""
		self._step = state['step']
		self._ep_idx = state['ep_idx']
		self._grad_step = state['grad_step']
		self._start_time = time() - state['elapsed_time']
		self.agent.load_training_state(state['agent'])
		self._buffer_chunks = state['buffer_chunks']
		self.buffer.restore(resume.buffer_directory(self.cfg), self._buffer_chunks, state['num_eps'])
		# A freshly built env reseeds its first reset with cfg.seed, which would replay the run's
		# first episodes; derive a seed unique to this run and step instead.
		if hasattr(self.env, 'seed_next_reset'):
			self.env.seed_next_reset(int(np.random.SeedSequence([self.cfg.seed, self._step]).generate_state(1)[0]))
		else:
			print(colored(f'{self.cfg.task} cannot be reseeded, so the resumed run replays its first episodes.', 'red'))
		random.setstate(state['rng']['python'])
		np.random.set_state(state['rng']['numpy'])
		torch.set_rng_state(state['rng']['torch'])
		torch.cuda.set_rng_state(state['rng']['cuda'], self.cfg.device)
		self._resumed = True
		print(colored(f'Resumed at step {self._step:,}.', 'yellow', attrs=['bold']))

	def to_td(self, obs, action=None, reward=None, terminated=None):
		"""Creates a TensorDict for a new episode."""
		if isinstance(obs, dict):
			# multi-stream observation (obs='rgb_state'). Each stream gets the same leading axis
			# the single-stream branch adds, so the nested batch size matches the (1,) below --
			# a sub-TensorDict whose batch size does not start with its parent's is rejected.
			obs = TensorDict(
				{k: v.unsqueeze(0).cpu() for k, v in obs.items()}, batch_size=(1,), device='cpu')
		else:
			obs = obs.unsqueeze(0).cpu()
		if action is None:
			action = torch.full_like(self.env.rand_act(), float('nan'))
		if reward is None:
			reward = torch.tensor(float('nan'))
		if terminated is None:
			terminated = torch.tensor(float('nan'))
		td = TensorDict(
			obs=obs,
			action=action.unsqueeze(0),
			reward=reward.unsqueeze(0),
			terminated=terminated.unsqueeze(0),
		batch_size=(1,))
		return td

	def train(self):
		"""Train a TD-MPC2 agent."""
		train_metrics, done, eval_next, backup_next = {}, True, False, False
		# A resumed run starts at the episode boundary its backup was taken at, where that step's eval
		# and the finished episode's buffer add have already happened, so the first boundary only resets.
		resumed, self._resumed = self._resumed, False
		while self._step <= self.cfg.steps:
			# Evaluate agent periodically
			if self._step % self.cfg.eval_freq == 0 and not resumed:
				eval_next = True

			# Reset environment
			if done:
				if not resumed:
					if eval_next:
						eval_metrics = self.eval()
						eval_metrics.update(self.common_metrics())
						self.logger.log(eval_metrics, 'eval')
						eval_next = False
						# not at step 0, whose eval precedes any data
						backup_next = self.cfg.save_resume_state and self._step > 0

					if self._step > 0:
						if info['terminated'] and not self.cfg.episodic:
							raise ValueError('Termination detected but you are not in episodic mode. ' \
							'Set `episodic=true` to enable support for terminations.')
						train_metrics.update(
							episode_reward=torch.tensor([td['reward'] for td in self._tds[1:]]).sum(),
							episode_success=info['success'],
							episode_length=len(self._tds),
							episode_terminated=info['terminated'])
						train_metrics.update(self.common_metrics())
						self.logger.log(train_metrics, 'train')
						self._ep_idx = self.buffer.add(torch.cat(self._tds))

					if backup_next:
						self.save_resume_state()
						backup_next = False

				resumed = False
				obs = self.env.reset()
				self._tds = [self.to_td(obs)]

			# Collect experience
			if self._step > self.cfg.seed_steps:
				action = self.agent.act(obs, t0=len(self._tds)==1)
			else:
				action = self.env.rand_act()
			obs, reward, done, info = self.env.step(action)
			self._tds.append(self.to_td(obs, action, reward, info['terminated']))

			# Update agent
			if self._step >= self.cfg.seed_steps:
				if self._step == self.cfg.seed_steps:
					num_updates = self.cfg.seed_steps
					print('Pretraining agent on seed data...')
				else:
					num_updates = 1
				for _ in range(num_updates):
					_train_metrics = self.agent.update(self.buffer)
				train_metrics.update(_train_metrics)

				# Periodic checkpoint, keyed on gradient updates rather than env steps since
				# seed pretraining performs `seed_steps` updates in one shot.
				prev_grad_step = self._grad_step
				self._grad_step += num_updates
				if self._grad_step // CHECKPOINT_FREQ > prev_grad_step // CHECKPOINT_FREQ:
					checkpoint_step = (self._grad_step // CHECKPOINT_FREQ) * CHECKPOINT_FREQ
					self.logger.save_agent(self.agent, identifier=f'epoch_{checkpoint_step}')

			self._step += 1

		self.logger.finish(self.agent)
		# the run is complete, so its backup (about one replay buffer of disk) is no longer needed
		resume.remove(self.cfg)
