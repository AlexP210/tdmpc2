import os
import shutil

import torch
from tensordict.tensordict import TensorDict
from torchrl.data.replay_buffers import ReplayBuffer, LazyTensorStorage
from torchrl.data.replay_buffers.samplers import SliceSampler

# Transitions copied at a time between the storage and a backup chunk, so neither direction holds a
# second in-memory copy of a whole chunk (4.5 GB per 10k transitions at camera_resolution=224)
CHUNK_SLICE = 1024


class Buffer():
	"""
	Replay buffer for TD-MPC2 training. Based on torchrl.
	Uses CUDA memory if available, and CPU memory otherwise.
	"""

	def __init__(self, cfg):
		self.cfg = cfg
		self._device = torch.device(cfg.device)
		self._capacity = min(cfg.buffer_size, cfg.steps)
		self._sampler = SliceSampler(
			num_slices=self.cfg.batch_size,
			end_key=None,
			traj_key='episode',
			truncated_key=None,
			strict_length=True,
			cache_values=cfg.multitask,
		)
		self._batch_size = cfg.batch_size * (cfg.horizon+1)
		self._num_eps = 0
		# storage indices written by `add` since the last `save_new`, oldest first
		self._unsaved = []

	@property
	def capacity(self):
		"""Return the capacity of the buffer."""
		return self._capacity

	@property
	def num_eps(self):
		"""Return the number of episodes in the buffer."""
		return self._num_eps

	def _reserve_buffer(self, storage):
		"""
		Reserve a buffer with the given storage.
		"""
		return ReplayBuffer(
			storage=storage,
			sampler=self._sampler,
			pin_memory=False,
			prefetch=0,
			batch_size=self._batch_size,
		)

	def _init(self, tds):
		"""Initialize the replay buffer. Use the first episode to estimate storage requirements."""
		print(f'Buffer capacity: {self._capacity:,}')
		mem_free, _ = torch.cuda.mem_get_info(device=self.cfg.device)
		bytes_per_step = sum([
				(v.numel()*v.element_size() if not isinstance(v, TensorDict) \
				else sum([x.numel()*x.element_size() for x in v.values()])) \
			for v in tds.values()
		]) / len(tds)
		total_bytes = bytes_per_step*self._capacity
		print(f'Storage required: {total_bytes/1e9:.2f} GB')
		# Heuristic: decide whether to use CUDA or CPU memory
		storage_device = self.cfg.device if 2.5*total_bytes < mem_free else 'cpu'
		print(f'Using {storage_device.upper()} memory for storage.')
		self._storage_device = torch.device(storage_device)
		return self._reserve_buffer(
			LazyTensorStorage(self._capacity, device=self._storage_device)
		)

	def load(self, td):
		"""
		Load a batch of episodes into the buffer. This is useful for loading data from disk,
		and is more efficient than adding episodes one by one.
		"""
		num_new_eps = len(td)
		episode_idx = torch.arange(self._num_eps, self._num_eps+num_new_eps, dtype=torch.int64)
		td['episode'] = episode_idx.unsqueeze(-1).expand(-1, td['reward'].shape[1])
		if self._num_eps == 0:
			self._buffer = self._init(td[0])
		td = td.reshape(td.shape[0]*td.shape[1])
		self._buffer.extend(td)
		self._num_eps += num_new_eps
		return self._num_eps

	def add(self, td):
		"""Add an episode to the buffer."""
		td['episode'] = torch.full_like(td['reward'], self._num_eps, dtype=torch.int64)
		if self._num_eps == 0:
			self._buffer = self._init(td)
		self._unsaved.append(self._buffer.extend(td))
		self._num_eps += 1
		return self._num_eps

	def save_new(self, directory):
		"""Write the transitions added since the last call to a new chunk under `directory`.

		Returns the chunk's `name` and `length`, or None if nothing was added. Only the newest
		`capacity` of them are still in storage, which is all `restore` needs anyway. The chunk is
		written under a temporary name and renamed once complete, so a crash mid-write leaves
		nothing `restore` would read.
		"""
		if not self._unsaved:
			return None
		idx = torch.cat(self._unsaved)[-self._capacity:]
		name = f'chunk_{self._num_eps:09d}'
		tmp = directory / f'{name}.tmp'
		shutil.rmtree(tmp, ignore_errors=True)
		storage = self._buffer.storage
		chunk = storage.get(idx[:1]).cpu().expand(len(idx)).memmap_like(tmp)
		for start in range(0, len(idx), CHUNK_SLICE):
			chunk[start:start+CHUNK_SLICE] = storage.get(idx[start:start+CHUNK_SLICE]).cpu()
		os.replace(tmp, directory / name)
		self._unsaved = []
		return dict(name=name, length=len(idx))

	def restore(self, directory, chunks, num_eps):
		"""Refill an empty buffer from the chunks `save_new` wrote, oldest first.

		Re-extending in the original order leaves the storage holding the same newest `capacity`
		transitions, with the oldest at the write cursor, so FIFO eviction carries on as it would
		have. `num_eps` is the episode count at the backup, which numbers the next episode added.
		"""
		assert self._num_eps == 0, 'restore needs an empty buffer'
		for info in chunks:
			chunk = TensorDict.load_memmap(directory / info['name'])
			for start in range(0, len(chunk), CHUNK_SLICE):
				part = chunk[start:start+CHUNK_SLICE]
				if not hasattr(self, '_buffer'):
					self._buffer = self._init(part)
				self._buffer.extend(part)
		self._num_eps = num_eps
		print(f'Restored {len(self._buffer):,} transitions from {len(chunks)} chunks.')

	def _prepare_batch(self, td):
		"""
		Prepare a sampled batch for training (post-processing).
		Expects `td` to be a TensorDict with batch size TxB.
		"""
		td = td.select("obs", "action", "reward", "terminated", "task", strict=False).to(self._device, non_blocking=True)
		obs = td.get('obs').contiguous()
		action = td.get('action')[1:].contiguous()
		reward = td.get('reward')[1:].unsqueeze(-1).contiguous()
		terminated = td.get('terminated', None)
		if terminated is not None:
			terminated = td.get('terminated')[1:].unsqueeze(-1).contiguous()
		else:
			terminated = torch.zeros_like(reward)
		task = td.get('task', None)
		if task is not None:
			task = task[0].contiguous()
		return obs, action, reward, terminated, task

	def sample(self):
		"""Sample a batch of subsequences from the buffer."""
		td = self._buffer.sample().view(-1, self.cfg.horizon+1).permute(1, 0)
		return self._prepare_batch(td)
