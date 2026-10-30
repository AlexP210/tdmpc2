"""The on-disk backup an online run resumes from: `<work_dir>/resume`.

It holds `state.pt` (everything but the replay buffer) and `buffer/`, the chunks
`Buffer.save_new` writes. `state.pt` lists the chunks that make up the buffer and is replaced
atomically, last, so a crash at any point of a backup leaves the previous one intact: a chunk not
yet listed is ignored and removed by the next backup.
"""

import os
import shutil
from pathlib import Path

import torch

STATE_FILE = 'state.pt'


def directory(cfg):
	return Path(cfg.work_dir) / 'resume'


def buffer_directory(cfg):
	path = directory(cfg) / 'buffer'
	path.mkdir(parents=True, exist_ok=True)
	return path


def exists(cfg):
	return (directory(cfg) / STATE_FILE).exists()


def save(cfg, state, chunks):
	"""Atomically replace `state.pt`, then drop the chunks it no longer lists."""
	path = directory(cfg) / STATE_FILE
	tmp = path.with_suffix('.tmp')
	torch.save(state, tmp)
	os.replace(tmp, path)
	keep = {chunk['name'] for chunk in chunks}
	for entry in buffer_directory(cfg).iterdir():
		if entry.name not in keep:
			shutil.rmtree(entry)


def load(cfg):
	# CPU, whatever device it was saved from: `load_state_dict` moves each tensor to its parameter
	return torch.load(directory(cfg) / STATE_FILE, map_location='cpu', weights_only=False)


def remove(cfg):
	shutil.rmtree(directory(cfg), ignore_errors=True)
