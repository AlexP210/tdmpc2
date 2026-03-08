import torch
import torch.nn.functional as F
import torch.nn as nn
from tdmpc2.common import math
from tdmpc2.common.scale import RunningScale
from tdmpc2.common.world_model import WorldModel
from tdmpc2.common.layers import api_model_conversion

from tensordict import TensorDict

from itertools import combinations
import random

class TDMPC2(torch.nn.Module):
	"""
	TD-MPC2 agent. Implements training + inference.
	Can be used for both single-task and supports both state and pixel observations.
	"""

	def __init__(self, cfg):
		super().__init__()
		self.cfg = cfg
		self.device = torch.device(self.cfg.device)
		self.model = WorldModel(cfg).to(self.device)
		self.optim = torch.optim.Adam([
			{'params': self.model._encoder.parameters(), 'lr': self.cfg.lr*self.cfg.enc_lr_scale},
			{'params': self.model._dynamics.parameters()},
			{'params': self.model._reward.parameters() if self.cfg.train_reward else []},
			{'params': self.model._termination.parameters() if self.cfg.episodic and self.cfg.train_rl else [] , 'lr': self.cfg.lr / self.cfg.num_r_d},
			{'params': self.model._Qs.parameters() if self.cfg.train_rl else [], 'lr': self.cfg.lr / self.cfg.num_r_d},
			{'params': []
			 }
		], lr=self.cfg.lr, capturable=True)
		if self.cfg.train_rl:
			self.pi_optim = torch.optim.Adam(self.model._pi.parameters(), lr=self.cfg.lr, eps=1e-5, capturable=True)
		self.model.eval()
		self.scale = RunningScale(cfg)
		self.cfg.iterations += 2*int(cfg.action_dim >= 20) # Heuristic for large action spaces
		self.discount =  self._get_discount(cfg.episode_length) # TODO
		# print('Episode length:', cfg.episode_length)
		# print('Discount factor:', self.discount)
		self._prev_mean = torch.nn.Buffer(torch.zeros(self.cfg.horizon, self.cfg.action_dim, device=self.device))
		if cfg.compile:
			print('Compiling update function with torch.compile...')
			self._update = torch.compile(self._update, mode="reduce-overhead")

	@property
	def plan(self):
		_plan_val = getattr(self, "_plan_val", None)
		if _plan_val is not None:
			return _plan_val
		if self.cfg.compile:
			plan = torch.compile(self._plan, mode="reduce-overhead")
		else:
			plan = self._plan
		self._plan_val = plan
		return self._plan_val

	def _get_discount(self, episode_length):
		"""
		Returns discount factor for a given episode length.
		Simple heuristic that scales discount linearly with episode length.
		Default values should work well for most tasks, but can be changed as needed.

		Args:
			episode_length (int): Length of the episode. Assumes episodes are of fixed length.

		Returns:
			float: Discount factor for the task.
		"""
		frac = episode_length/self.cfg.discount_denom
		return min(max((frac-1)/(frac), self.cfg.discount_min), self.cfg.discount_max)

	def save(self, fp):
		"""
		Save state dict of the agent to filepath.

		Args:
			fp (str): Filepath to save state dict to.
		"""
		torch.save({"model": self.model.state_dict(), "optim":self.optim.state_dict()}, fp)
	
	def load(self, fp):
		"""
		Load a saved state dict from filepath (or dictionary) into current agent.

		Args:
			fp (str or dict): Filepath or state dict to load.
		"""
		if isinstance(fp, dict):
			state_dict = fp
		else:
			state_dict = torch.load(fp, map_location=torch.get_default_device(), weights_only=False)
		state_dict = state_dict["model"] if "model" in state_dict else state_dict
		state_dict = api_model_conversion(self.model.state_dict(), state_dict)
		self.model.load_state_dict(state_dict)
		return

	# @torch.no_grad()
	# def rand_act(self, obs, env, eval_mode=False):
	# 	"""
	# 	Plan a sequence of actions using the learned world model.

	# 	Args:
	# 		obs (torch.Tensor): Observation from environment.
	# 		env: Environment instance.
	# 		eval_mode (bool): If True, turn off uncertainty bonuses.

	# 	Returns:
	# 		action (torch.Tensor): Action to take (shape: [action_dim]).
	# 		info (dict): Dictionary of extra outputs like reward and uncertainty terms.
	# 	"""
	# 	obs = obs.to(self.device, non_blocking=True).unsqueeze(0)  # [1, obs_dim]
	# 	action = env.rand_act()
	# 	action = torch.tensor(action, dtype=torch.float32, device=self.device).unsqueeze(0)  # [1, act_dim]

	# 	# Encode observation
	# 	z = self.model.encode(obs)  # [1, latent_dim]

	# 	# Reward prediction
	# 	reward = self.model.reward(z, action)

	# 	# Transition prediction
	# 	z, dyn_epi_uncer = self.model.next(z, action, return_epistemic=True)

	# 	# Uncertainty weighting (disabled during eval)
	# 	dyn_beta = self.cfg.dyn_uncer_beta_coef

	# 	adjusted_reward = (1-dyn_beta)*reward + dyn_beta*dyn_epi_uncer

	# 	# Collect info
	# 	info = {
	# 		"value" : 0 ,
	# 		"reward": reward.squeeze(0),                     # scalar
	# 		"dyn_epistemic": dyn_epi_uncer.squeeze(0),       # scalar
	# 		"exploration_reward": adjusted_reward.squeeze(0),   # scalar
	# 	}
	# 	#TODO: Nitpicking value is not calculated for rand action, and since ubp does not use value, it is not needed now but it would be a nice plot to have 
	# 	return action.clamp(-1, 1).squeeze(0), info

	@torch.no_grad()
	def act(self, obs, t0=False, eval_mode=False):
		"""
		Select an action by planning in the latent space of the world model.

		Args:
			obs (torch.Tensor): Observation from the environment.
			t0 (bool): Whether this is the first observation in the episode.
			eval_mode (bool): Whether to use the mean of the action distribution.

		Returns:
			torch.Tensor: Action to take in the environment.
		"""
		obs = obs.to(self.device, non_blocking=True).unsqueeze(0)
		if self.cfg.mpc:
			return self.plan(obs, t0=t0, eval_mode=eval_mode)
		if not self.cfg.train_rl:
			raise NotImplementedError("cfg.train_rl and cfg.mpc both set to `False`.")
		z = self.model.encode(obs)
		action, info = self.model.pi(z)
		if eval_mode:
			action = info["mean"]
		return action[0].cpu()

	@torch.no_grad()
	def _estimate_value(self, z, actions, eval_mode=False):
		"""Estimate value of a trajectory starting at latent state z and executing given actions.
		eval: N = number of samples 512, num_ensemble = 5 , D = 512
			z = [512,512] [N , D]
			actions = [3,512,4] [T, N , A]
			eval_mode = True
		"""

		# Accumulators for summing over time
		termination = torch.zeros(self.cfg.num_samples, 1, dtype=torch.float32, device=z.device)
		G          = torch.zeros_like(z[:, :1])      # [N,1]
		R          = torch.zeros_like(z[:, :1])
		discount   = torch.ones(1, device=z.device)  # scalar tensor
		epi_dyn    = torch.zeros_like(z[:, :1])

		for t in range(self.cfg.horizon):
			# reward_ens = math.two_hot_inv(reward_ens, self.cfg) removed because the preference model is deterministic
			# Reward prediction: reward =  [N , 1] , reward_epi_uncer = [N , 1] , reward_aleatoric_uncer = [N, 1]
			if self.cfg.train_reward:
				extrinsic_reward = math.two_hot_inv(self.model.reward(z, actions[t]), self.cfg)
			else:
				extrinsic_reward = 0

			# Dynamics prediction
			# Next State prediction: reward =  [N , D] , reward_epi_uncer = [N , 1] , reward_aleatoric_uncer = [N, 1]
			z, dyn_epi_uncer = self.model.next(z, actions[t], return_epistemic=True)
			dyn_beta = self.cfg.dyn_uncer_beta_coef

			# Adjusted reward (reward bonus shaping)
			adjusted_reward = (1-dyn_beta) * extrinsic_reward + dyn_beta * dyn_epi_uncer
			G = G + discount * (1 - termination) * adjusted_reward

			# Discount update
			discount = discount * self.discount

			# Termination update
			if self.cfg.episodic and self.cfg.train_rl:
				termination = torch.clip(termination + (self.model.termination(z) > 0.5).float(), max=1.)

			# Sum all quantities over time [N,1]
			R = R + extrinsic_reward
			epi_dyn += dyn_epi_uncer

		# Bootstrap value from final state
		if self.cfg.train_rl:
			action, _ = self.model.pi(z)
			value = G + discount * (1 - termination) * self.model.Q(z, action, return_type='avg')
		else:
			value = G
		# Info for logging
		info = {
			"total_return" : G,
			"dynamics_epistemic": epi_dyn,
		}
		value = value.nan_to_num(0)
		return value, info

	@torch.no_grad()
	def _plan(self, obs, t0=False, eval_mode=False):
		"""
		Plan a sequence of actions using the learned world model.

		Args:
			obs(torch.Tensor): state from which to plan. [1,39]
			t0 (bool): Whether this is the first observation in the episode.
			eval_mode (bool): Whether to use the uncertainty dynamics in planning, if true , uncertainty is not used

		Returns:
			torch.Tensor: Action to take in the environment.
		"""
		# Latent Space encoding for the current observation [1,Latent Dimension D]
		z = self.model.encode(obs)
		# Sample policy trajectories [24]
		if self.cfg.num_pi_trajs > 0:
			if not self.cfg.train_rl:
				raise NotImplementedError("cfg.num_pi_trajs > 0 but cfg.train_rl = False")
			pi_actions = torch.empty(self.cfg.horizon, self.cfg.num_pi_trajs, self.cfg.action_dim, device=self.device)
			# Repeated State [self.cfg.num_pi_trajs, D]
			_z = z.repeat(self.cfg.num_pi_trajs, 1)
			# Actions sampled from policy [T,self.cfg.num_pi_trajs,A]
			for t in range(self.cfg.horizon-1):
				pi_actions[t], _ = self.model.pi(_z)
				_z = self.model.next(_z, pi_actions[t])
			pi_actions[-1], _ = self.model.pi(_z)

		# Initialize state and parameters
		# Repeated State [N, D]
		z = z.repeat(self.cfg.num_samples, 1)
		#Mean for action [T, A]
		mean = torch.zeros(self.cfg.horizon, self.cfg.action_dim, device=self.device)
		#std for action [T, A]
		std = torch.full((self.cfg.horizon, self.cfg.action_dim), self.cfg.max_std, dtype=torch.float, device=self.device)
		if not t0:
			mean[:-1] = self._prev_mean[1:]
		
		#Actions  are [T, N , A] filling the first self.cfg.num_pi_trajs number of them with [T, self.cfg.num_pi_trajs , A]
		actions = torch.empty(self.cfg.horizon, self.cfg.num_samples, self.cfg.action_dim, device=self.device)
		if self.cfg.num_pi_trajs > 0:
			actions[:, :self.cfg.num_pi_trajs] = pi_actions

		# Iterate MPPI
		for _ in range(self.cfg.iterations):

			# Sample actions for non policy sampled actions (empty ones)
			r = torch.randn(self.cfg.horizon, self.cfg.num_samples-self.cfg.num_pi_trajs, self.cfg.action_dim, device=std.device)
			actions_sample = mean.unsqueeze(1) + std.unsqueeze(1) * r
			actions_sample = actions_sample.clamp(-1, 1)
			actions[:, self.cfg.num_pi_trajs:] = actions_sample

			# Compute value, reward, and associated uncertainty info 
			# value : [N,1] , ubp_reward[N,1] , infos [N, 1]
			value, info = self._estimate_value(z, actions, eval_mode)
			

			# Define metrics with default fallback to "ubp_reward"
			planning_metric_map = {
				"total_value": value.squeeze(1),
			}
			metric_key = getattr(self.cfg, "planning_criteria", "total_value")
			metric_values = planning_metric_map.get(metric_key, planning_metric_map["total_value"]) #the ubp reward [N,1]

			# Top-k selection [64]
			elite_idxs = torch.topk(metric_values, self.cfg.num_elites, dim=0).indices  # [num_elites]

			# Extract elite values and actions
			elite_value = value[elite_idxs]                   # [num_elites, 1]
			elite_actions = actions[:, elite_idxs]            # [horizon, num_elites, action_dim]

			# Extract elite info tensors
			elite_info = {k: v[elite_idxs] for k, v in info.items()}  #[ num elites, 1]

			# Elite scoring
			elite_metric = metric_values[elite_idxs].unsqueeze(1)        # [num_elites, 1]
			max_metric = elite_metric.max(0).values
			score = torch.exp(self.cfg.temperature * (elite_metric - max_metric))  # [num_elites, 1]
			score = score / score.sum(0)
			mean = (score.unsqueeze(0) * elite_actions).sum(dim=1) / (score.sum(0) + 1e-9) #[1, num elite, 1] * [T,num elite,A] = weighted actinos by score [T, num elite, A] , sum dim 1 : [T,A] , divide by scaler , mean is T,A
			std = ((score.unsqueeze(0) * (elite_actions - mean.unsqueeze(1)) ** 2).sum(dim=1) / (score.sum(0) + 1e-9)).sqrt()
			std = std.clamp(self.cfg.min_std, self.cfg.max_std) 							#[T,A]

		# Select action
		rand_idx = math.gumbel_softmax_sample(score.squeeze(1)) #scaler
		actions = torch.index_select(elite_actions, 1, rand_idx).squeeze(1) #[T,A]
		# Final info dict
		info = {
			"value":  elite_value[rand_idx].squeeze(0), #value of the random action chosen from the elite actions, scaler 
			"total_return": elite_info["total_return"][rand_idx].squeeze(0), #reward of the random action chosen from the elite actions, scaler 
			"dyn_epistemic": elite_info["dynamics_epistemic"][rand_idx].squeeze(0), #dyn epi uncertainty of the random action chosen from the elite actions, scaler
		}
		a, std= actions[0], std[0]
		if not eval_mode:
			a = a + std * torch.randn(self.cfg.action_dim, device=std.device)
		self._prev_mean.copy_(mean)
		return a.clamp(-1, 1), info

	def update_pi(self, zs):
		"""
		Update policy using a sequence of latent states.

		Args:
			zs (torch.Tensor): Sequence of latent states.

		Returns:
			float: Loss of the policy update.
		"""
		action, info = self.model.pi(zs)
		qs = self.model.Q(zs, action, return_type='avg', detach=True)
		self.scale.update(qs[0])
		qs = self.scale(qs)

		# Loss is a weighted sum of Q-values
		rho = torch.pow(self.cfg.rho, torch.arange(len(qs), device=self.device))
		pi_loss = (-(self.cfg.entropy_coef * info["scaled_entropy"] + qs).mean(dim=(1,2)) * rho).mean()
		pi_loss.backward()
		pi_grad_norm = torch.nn.utils.clip_grad_norm_(self.model._pi.parameters(), self.cfg.grad_clip_norm)
		self.pi_optim.step()
		self.pi_optim.zero_grad(set_to_none=True)

		info = TensorDict({
			"pi_loss": pi_loss,
			"pi_grad_norm": pi_grad_norm,
			"pi_entropy": info["entropy"],
			"pi_scaled_entropy": info["scaled_entropy"],
			"pi_scale": self.scale.value,
		})
		return info

	@torch.no_grad()
	def _td_target(self, next_z, reward, terminated):
		"""
		Compute the TD-target from a reward and the observation at the following time step.

		Args:
			next_z (torch.Tensor): Latent state at the following time step.
			reward (torch.Tensor): Reward at the current time step.
			terminated (torch.Tensor): Termination signal at the current time step.

		Returns:
			torch.Tensor: TD-target.
		"""
		action, _ = self.model.pi(next_z)
		return reward + self.discount * (1-terminated) * self.model.Q(next_z, action, return_type='min', target=True)

	def _update_independant(self, observations, actions, extrinsic_rewards, terminateds):
		"""
		# Updating the loss is not dependant on episetmic uncertainty, the variance of each single member is only used for its loss computation only
		obs: A batch of [batch size], each element in the batch is four conseqtive observations (4 because T = 3 +1), each observation is obs dimension [T+1,B,39]
		action: A batch of [batch size], each element in the batch is three conseqtive actions (because T = 3), each action is action dimension [T,B,4]
		reward: A batch of [batch size], each element in the batch is three conseqtive rewards (because T = 3), each reward is scaler [T,B,1]
		terminated: A batch of [batch size], each element in the batch is three conseqtive terminated status (because T = 3), each reward is scaler [T,B,1]
		"""
		consistency_loss_all, reward_loss_all, value_loss_all, termination_loss_all, total_loss_all, grad_norm_all = 0,0,0,0,0,0
		epistemic_uncertainty = 0

		for member in range(self.cfg.num_r_d):
			obs = observations[member]
			action = actions[member]
			extrinsic_reward = extrinsic_rewards[member]
			terminated = terminateds[member]
			# Compute the various quantities we need
			with torch.no_grad():
				# Next latents, present latents
				next_z_true = self.model.encode(obs[1:]) # [T,B, D] the latent dimension of the next states of the current observation in the selected sequence [observation at T=2,3,4] ie target next states for the current state
				z = self.model.encode(obs[:-1])

				# Intrinsic rewards
				intrinsic_reward = torch.empty_like(extrinsic_reward)
				for (t, (_z, _a)) in enumerate(zip(z.unbind(0), action.unbind(0))):
					_, dyn_epi = self.model.next(_z, _a, return_epistemic=True)
					intrinsic_reward[t] = dyn_epi
				epistemic_uncertainty = intrinsic_reward.mean()

				# Total reward
				dyn_beta = self.cfg.dyn_uncer_beta_coef
				if self.cfg.train_reward:
					reward =  (1-dyn_beta) * extrinsic_reward + dyn_beta*intrinsic_reward
				else:
					reward =  dyn_beta*intrinsic_reward

				# TD targets (if needed)
				if self.cfg.train_rl:
					td_targets = self._td_target(next_z_true, reward, terminated) #[T,B,1] target expected return starting from current state


			# Prepare for update
			self.model.train()

			# Latent rollout
			# zs is the Predicted states, except for the first observation in the sequence because it is the first one, can not be predicetd given something. [4 (T+1), Batch Size , Latent Dimension]
			zs = torch.empty(self.cfg.horizon+1, self.cfg.batch_size, self.cfg.latent_dim, device=self.device)
			# encode the first observation of each sequence [B, D]
			z = self.model.encode(obs[0])
			# fill the latent state of the non predictable first observation
			zs[0] = z

			# Compute the Consistency loss
			consistency_loss = 0
			for t, (_action, _next_z) in enumerate(zip(action.unbind(0), next_z_true.unbind(0))):
				z = self.model.next_single_member(z, _action, member)  # mu shape  is [batch_size, latent_dim] , var shape is [batch size,latent dim]
				consistency_loss = consistency_loss + F.mse_loss(z, _next_z) * self.cfg.rho**t
				zs[t+1] = z #fill each predicted latent observation in zs
			consistency_loss = consistency_loss / self.cfg.horizon

			# Compute Value losses
			if self.cfg.train_rl:
				value_loss = 0
				_zs = zs[:-1] #[T,B , D]
				qs = self.model.Q(_zs, action, return_type='all')
				for t, (td_targets_unbind, qs_unbind) in enumerate(zip(td_targets.unbind(0), qs.unbind(1))):
					for _, qs_unbind_unbind in enumerate(qs_unbind.unbind(0)):
						value_loss = value_loss + math.soft_ce(qs_unbind_unbind, td_targets_unbind, self.cfg).mean() * self.cfg.rho**t
				value_loss = value_loss / (self.cfg.horizon * self.cfg.num_q)
			else:
				value_loss = 0.

			# Compute Reward losses (if needed)
			if self.cfg.train_reward:
				reward_loss = 0
				reward_preds = self.model.reward(_zs, action)
				for t, (rew_pred_unbind, rew_unbind) in enumerate(zip(reward_preds.unbind(0), reward.unbind(0))):
					reward_loss = reward_loss + math.soft_ce(rew_pred_unbind, rew_unbind, self.cfg).mean() * self.cfg.rho**t
				reward_loss = reward_loss / self.cfg.horizon
			else:
				reward_loss = 0.

			# Compute the termination lossess (if needed)
			if self.cfg.episodic and self.cfg.train_rl:
				termination_pred = self.model.termination(zs[1:], unnormalized=True)
				termination_loss = F.binary_cross_entropy_with_logits(termination_pred, terminated)
			else:
				termination_loss = 0.

			total_loss = (
				self.cfg.consistency_coef * consistency_loss +
				self.cfg.reward_coef * reward_loss +
				self.cfg.termination_coef * termination_loss +
				self.cfg.value_coef * value_loss
			)

			# Update model
			total_loss.backward()
			grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip_norm)
			self.optim.step()
			self.optim.zero_grad(set_to_none=True)

			# Update policy & target Q functions
			if self.cfg.train_rl:
				pi_info = self.update_pi(zs.detach())
				self.model.soft_update_target_Q()

			consistency_loss_all =consistency_loss_all + consistency_loss 
			reward_loss_all = reward_loss_all + reward_loss 
			value_loss_all = value_loss_all + value_loss
			termination_loss_all = termination_loss_all + termination_loss 
			total_loss_all = total_loss_all + total_loss
			grad_norm_all = grad_norm_all + grad_norm 
		
		# Return training statistics
		self.model.eval()
		info = TensorDict({
			"consistency_loss": consistency_loss_all / self.cfg.num_r_d,
			"value_loss": value_loss_all / self.cfg.num_r_d,
			"termination_loss": termination_loss_all / self.cfg.num_r_d,
			"total_loss": total_loss_all / self.cfg.num_r_d,
			"grad_norm": grad_norm_all / self.cfg.num_r_d,
			"epistemic_uncertainty": epistemic_uncertainty 
		})
		if self.cfg.train_reward:
			info.update(TensorDict({"reward_loss": reward_loss_all / self.cfg.num_r_d}))
		if self.cfg.episodic and self.cfg.train_rl:
			info.update(math.termination_statistics(torch.sigmoid(termination_pred[-1]), terminated[-1]))
		if self.cfg.train_rl:
			info.update(pi_info)
		return info.detach().mean()

	def update(self, buffer):
		"""
		Main update function. Corresponds to one iteration of model learning.

		Args:
			buffer (common.buffer.Buffer): Replay buffer.
			add_pref (bool): Whether to add preference feedback this update based on pref feedback schedule 

		Returns:
			dict: Dictionary of training statistics.
		"""
		observations = []
		states = []
		actions = []
		extrinsic_rewards = []
		terminateds = []
		for _ in range(self.cfg.num_r_d):
			obs, state, action, extrinsic_reward, terminated = buffer.sample()	
			observations.append(obs)
			states.append(state)
			actions.append(action)
			extrinsic_rewards.append(extrinsic_reward)
			terminateds.append(terminated)
		torch.compiler.cudagraph_mark_step_begin()
		return self._update_independant(observations, actions, extrinsic_rewards, terminateds)