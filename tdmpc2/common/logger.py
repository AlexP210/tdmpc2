import dataclasses
import os
import datetime
import re

import numpy as np
import pandas as pd
import h5py
from termcolor import colored

from tdmpc2.common import TASK_SET


CONSOLE_FORMAT = [
	("iteration", "I", "int"),
	("episode", "E", "int"),
	("step", "I", "int"),
	("episode_reward", "R", "float"),
	("episode_success", "S", "float"),
	("elapsed_time", "T", "time"),
]

CAT_TO_COLOR = {
	"pretrain": "yellow",
	"train": "blue",
	"eval": "green",
}


def make_dir(dir_path):
	"""Create directory if it does not already exist."""
	try:
		os.makedirs(dir_path)
	except OSError:
		pass
	return dir_path


def print_run(cfg):
	"""
	Pretty-printing of current run information.
	Logger calls this method at initialization.
	"""
	prefix, color, attrs = "  ", "green", ["bold"]

	def _limstr(s, maxlen=36):
		return str(s[:maxlen]) + "..." if len(str(s)) > maxlen else s

	def _pprint(k, v):
		print(
			prefix + colored(f'{k.capitalize()+":":<15}', color, attrs=attrs), _limstr(v)
		)

	observations  = ", ".join([str(v) for v in cfg.obs_shape.values()])
	kvs = [
		("task", cfg.task_title),
		("steps", f"{int(cfg.steps):,}"),
		("observations", observations),
		("actions", cfg.action_dim),
		("experiment", cfg.exp_name),
	]
	w = np.max([len(_limstr(str(kv[1]))) for kv in kvs]) + 25
	div = "-" * w
	print(div)
	for k, v in kvs:
		_pprint(k, v)
	print(div)


def cfg_to_group(cfg, return_list=False):
	"""
	Return a wandb-safe group name for logging.
	Optionally returns group name as list.
	"""
	lst = [cfg.task, re.sub("[^0-9a-zA-Z]+", "-", cfg.exp_name)]
	return lst if return_list else "-".join(lst)


class VideoRecorder:
	"""Utility class for logging evaluation videos."""

	def __init__(self, cfg, wandb, fps=15):
		self.cfg = cfg
		self._save_dir = make_dir(os.path.join(cfg.work_dir, 'eval_video'))
		self._wandb = wandb
		self.fps = fps
		self.frames = []
		self.enabled = False

	def init(self, env, enabled=True):
		self.frames = []
		self.enabled = self._save_dir and self._wandb and enabled
		self.record(env)

	def record(self, env):
		if self.enabled:
			self.frames.append(env.render())

	def save(self, step, key='videos/eval_video'):
		if self.enabled and len(self.frames) > 0:
			frames = np.stack(self.frames)
			return self._wandb.log(
				{key: self._wandb.Video(frames.transpose(0, 3, 1, 2), fps=self.fps, format='mp4')}, step=step
			)


class Logger:
	"""Primary logging object. Logs either locally or using wandb."""

	def __init__(self, cfg):
		self.cfg = cfg
		self._log_dir = make_dir(cfg.work_dir)
		self._model_dir = make_dir(os.path.join(self._log_dir, "models"))
		self._save_csv = cfg.save_csv
		self._save_agent = cfg.save_agent
		self._group = cfg_to_group(cfg)
		self._seed = cfg.seed
		self._eval = []
		print_run(cfg)
		self.project = cfg.get("wandb_project", "none")
		self.entity = cfg.get("wandb_entity", "none")

		self.save_training_data = cfg.save_training_data
		self._training_dataset = None
		if self.save_training_data > 0:
			obs_dim = self.cfg.obs_shape[self.cfg.obs]
			save_dir = os.path.join(self._log_dir, "data")
			make_dir(save_dir)
			save_path = os.path.join(save_dir, "training.h5")
			self._training_dataset = self.create_transition_dataset(path=save_path)
			self._transition_buffer_size = 1024
			self._data_save_index = 0
			self.o_buffer = np.empty(shape=(self._transition_buffer_size, *obs_dim), dtype=float)
			self.a_buffer = np.empty(shape=(self._transition_buffer_size, cfg.action_dim), dtype=float)
			self.r_buffer = np.empty(shape=(self._transition_buffer_size, 1), dtype=float)
			self.oprime_buffer = np.empty(shape=(self._transition_buffer_size, *obs_dim), dtype=float)
			self.termination_buffer = np.empty(shape=(self._transition_buffer_size, 1), dtype=bool)
			self.truncation_buffer = np.empty(shape=(self._transition_buffer_size, 1), dtype=bool)
			self.state_buffer = np.empty(shape=(self._transition_buffer_size, self.cfg.state_dim), dtype=float)
			self.state_prime_buffer = np.empty(shape=(self._transition_buffer_size, self.cfg.state_dim), dtype=float)

		if not cfg.enable_wandb or self.project == "none" or self.entity == "none":
			print(colored("Wandb disabled.", "blue", attrs=["bold"]))
			cfg.save_agent = False
			cfg.save_video = False
			self._wandb = None
			self._video = None
			return
		os.environ["WANDB_SILENT"] = "true" if cfg.wandb_silent else "false"
		import wandb

		agent, task, seed = self._log_dir.split(os.sep)[-4:-1]
		wandb.init(
			project=self.project,
			entity=self.entity,
			name=f"{agent} {task} {seed}",
			group=self._group,
			tags=cfg_to_group(cfg, return_list=True) + [f"seed:{cfg.seed}"],
			dir=self._log_dir,
			config=dataclasses.asdict(cfg),
		)
		print(colored("Logs will be synced with wandb.", "blue", attrs=["bold"]))
		self._wandb = wandb
		self._video = (
			VideoRecorder(cfg, self._wandb)
			if self._wandb and cfg.save_video
			else None
		)

	@property
	def video(self):
		return self._video

	@property
	def model_dir(self):
		return self._model_dir

	def save_agent(self, agent=None, identifier='final'):
		if self._save_agent and agent:
			fp = os.path.join(self._model_dir, f'{str(identifier)}.pt')
			agent.save(fp)
			# if self._wandb:
			# 	artifact = self._wandb.Artifact(
			# 		self._group + '-' + str(self._seed) + '-' + str(identifier),
			# 		type='model',
			# 	)
			# 	artifact.add_file(fp)
			# 	self._wandb.log_artifact(artifact)

	def finish(self, agent=None):
		if self.save_training_data > 0:
			self.flush_transition_data()
			final_size = self._transitions_collected
			for key in self._training_dataset:
				ds = self._training_dataset[key]
				ds.resize((final_size, *ds.shape[1:]))
			self._training_dataset.close()
		try:
			self.save_agent(agent)
		except Exception as e:
			print(colored(f"Failed to save model: {e}", "red"))
		if self._wandb:
			self._wandb.finish()

	def _format(self, key, value, ty):
		if ty == "int":
			return f'{colored(key+":", "blue")} {int(value):,}'
		elif ty == "float":
			return f'{colored(key+":", "blue")} {value:.01f}'
		elif ty == "time":
			value = str(datetime.timedelta(seconds=int(value)))
			return f'{colored(key+":", "blue")} {value}'
		else:
			raise f"invalid log format type: {ty}"

	def _print(self, d, category):
		category = colored(category, CAT_TO_COLOR[category])
		pieces = [f" {category:<14}"]
		for k, disp_k, ty in CONSOLE_FORMAT:
			if k in d:
				pieces.append(f"{self._format(disp_k, d[k], ty):<22}")
		print("   ".join(pieces))

	def pprint_multitask(self, d, cfg):
		"""Pretty-print evaluation metrics for multi-task training."""
		print(colored(f'Evaluated agent on {len(cfg.tasks)} tasks:', 'yellow', attrs=['bold']))
		dmcontrol_reward = []
		metaworld_reward = []
		metaworld_success = []
		for k, v in d.items():
			if '+' not in k:
				continue
			task = k.split('+')[1]
			if task in TASK_SET['mt30'] and k.startswith('episode_reward'): # DMControl
				dmcontrol_reward.append(v)
				print(colored(f'  {task:<22}\tR: {v:.01f}', 'yellow'))
			elif task in TASK_SET['mt80'] and task not in TASK_SET['mt30']: # Meta-World
				if k.startswith('episode_reward'):
					metaworld_reward.append(v)
				elif k.startswith('episode_success'):
					metaworld_success.append(v)
					print(colored(f'  {task:<22}\tS: {v:.02f}', 'yellow'))
		dmcontrol_reward = np.nanmean(dmcontrol_reward)
		d['episode_reward+avg_dmcontrol'] = dmcontrol_reward
		print(colored(f'  {"dmcontrol":<22}\tR: {dmcontrol_reward:.01f}', 'yellow', attrs=['bold']))
		if cfg.task == 'mt80':
			metaworld_reward = np.nanmean(metaworld_reward)
			metaworld_success = np.nanmean(metaworld_success)
			d['episode_reward+avg_metaworld'] = metaworld_reward
			d['episode_success+avg_metaworld'] = metaworld_success
			print(colored(f'  {"metaworld":<22}\tR: {metaworld_reward:.01f}', 'yellow', attrs=['bold']))
			print(colored(f'  {"metaworld":<22}\tS: {metaworld_success:.02f}', 'yellow', attrs=['bold']))

	def log(self, d, category="train"):
		assert category in CAT_TO_COLOR.keys(), f"invalid category: {category}"
		if self._wandb:
			if category in {"train", "eval"}:
				xkey = "step"
			elif category == "pretrain":
				xkey = "iteration"
			_d = dict()
			for k, v in d.items():
				_d[category + "/" + k] = v
			self._wandb.log(_d, step=d[xkey])
		if category == "eval" and self._save_csv:
			keys = ["step", "episode_reward"]
			self._eval.append(np.array([d[keys[0]], d[keys[1]]]))
			pd.DataFrame(np.array(self._eval)).to_csv(
				os.path.join(self._log_dir, "eval.csv"), header=keys, index=None
			)
		self._print(d, category)

	def create_transition_dataset(self, path):
		dataset = h5py.File(path, "w")
		obs_dim = self.cfg.obs_shape[self.cfg.obs]
		o_dataset = dataset.create_dataset(
			name="o", 
			shape=(self.cfg.steps, *obs_dim),
			maxshape=(None, *obs_dim),
			chunks=(1024, *obs_dim),
			dtype="float32",
			compression="gzip",
			compression_opts=4,
		)
		a_dataset = dataset.create_dataset(
			name="a", 
			shape=(self.cfg.steps, self.cfg.action_dim),
			maxshape=(None, self.cfg.action_dim),
			chunks=(1024, self.cfg.action_dim),
			dtype="float32",
			compression="gzip",
			compression_opts=4,
		)
		r_dataset = dataset.create_dataset(
			name="r", 
			shape=(self.cfg.steps, 1),
			maxshape=(None, 1),
			chunks=(1024, 1),
			dtype="float32",
			compression="gzip",
			compression_opts=4,
		)
		oprime_dataset = dataset.create_dataset(
			name="oprime", 
			shape=(self.cfg.steps, *obs_dim),
			maxshape=(None, *obs_dim),
			chunks=(1024, *obs_dim),
			dtype="float32",
			compression="gzip",
			compression_opts=4,
		)
		terminated_dataset = dataset.create_dataset(
			name="terminated", 
			shape=(self.cfg.steps, 1),
			maxshape=(None, 1),
			chunks=(1024, 1),
			dtype="bool",
			compression="gzip",
			compression_opts=4,
		)
		truncated_dataset = dataset.create_dataset(
			name="truncated", 
			shape=(self.cfg.steps, 1),
			maxshape=(None, 1),
			chunks=(1024, 1),
			dtype="bool",
			compression="gzip",
			compression_opts=4,
		)
		state_dataset = dataset.create_dataset(
			name="state", 
			shape=(self.cfg.steps, self.cfg.state_dim),
			maxshape=(None, self.cfg.state_dim),
			chunks=(1024, self.cfg.state_dim),
			dtype="float32",
			compression="gzip",
			compression_opts=4,
		)
		state_prime_dataset = dataset.create_dataset(
			name="state_prime", 
			shape=(self.cfg.steps, self.cfg.state_dim),
			maxshape=(None, self.cfg.state_dim),
			chunks=(1024, self.cfg.state_dim),
			dtype="float32",
			compression="gzip",
			compression_opts=4,
		)

		self._transitions_collected = 0
		self._transitions_buffered = 0
		return dataset

	def _ensure_capacity_for_transition_data(self):
		if self._transitions_collected + self._transitions_buffered < self._training_dataset["o"].shape[0]:
			return

		# grow geometrically
		old_size = self._training_dataset["o"].shape[0]
		new_size = int(old_size * 1.5)

		for key in self._training_dataset:
			ds = self._training_dataset[key]
			ds.resize((new_size, *ds.shape[1:]))

	def flush_transition_data(self):
		start = self._transitions_collected
		end = start + self._transitions_buffered
		self._ensure_capacity_for_transition_data()
		self._training_dataset["o"][start:end] = self.o_buffer[:self._transitions_buffered]
		self._training_dataset["a"][start:end] = self.a_buffer[:self._transitions_buffered]
		self._training_dataset["r"][start:end] = self.r_buffer[:self._transitions_buffered]
		self._training_dataset["oprime"][start:end] = self.oprime_buffer[:self._transitions_buffered]
		self._training_dataset["terminated"][start:end] = self.termination_buffer[:self._transitions_buffered]
		self._training_dataset["truncated"][start:end] = self.truncation_buffer[:self._transitions_buffered]
		self._training_dataset["state"][start:end] = self.state_buffer[:self._transitions_buffered]
		self._training_dataset["state_prime"][start:end] = self.state_prime_buffer[:self._transitions_buffered]
		self._transitions_collected += self._transitions_buffered
		self._transitions_buffered = 0

	def log_transition(self, o, a, r, o_prime, terminated, truncated, s, s_prime):
		if np.random.rand() <= self.save_training_data:
			self.o_buffer[self._transitions_buffered] = o.detach().cpu().numpy()
			self.a_buffer[self._transitions_buffered] = a.detach().cpu().numpy()
			self.r_buffer[self._transitions_buffered] = r.detach().cpu().numpy()
			self.oprime_buffer[self._transitions_buffered] = o_prime.detach().cpu().numpy()
			self.termination_buffer[self._transitions_buffered] = terminated.detach().cpu().numpy()
			self.truncation_buffer[self._transitions_buffered] = truncated.detach().cpu().numpy()
			self.state_buffer[self._transitions_buffered] = s.detach().cpu().numpy()
			self.state_prime_buffer[self._transitions_buffered] = s_prime.detach().cpu().numpy()
			self._transitions_buffered += 1

			if self._transitions_buffered == self._transition_buffer_size:
				self.flush_transition_data()