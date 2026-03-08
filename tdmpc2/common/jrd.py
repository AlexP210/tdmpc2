import torch
import numpy as np


class JensenRenyiDivergence():
    def __init__(self, states_mean, states_var):
        self.states_mean = states_mean.permute(1, 0, 2) #[B, E, D] [1,5,512]
        self.states_var = states_var.permute(1, 0, 2) #[B, E, D] [1,5,512]

    def compute_measure(self):
        state_delta_means = self.states_mean #[B, E, D] [1,5,512]
        state_delta_vars = self.states_var #[B, E, D] [1,5,512]

        # shape: both (n_actors, ensemble_size, d_state)
        mu, var = state_delta_means, state_delta_vars 

        n_act, es, d_s = mu.size()

        # entropy of the mean #[B, E, , E , D] [1,5,39]
        # shape: (n_actors, ensemble_size, ensemble_size, d_state)
        mu_diff = mu.unsqueeze(1) - mu.unsqueeze(2)
        # shape: (n_actors, ensemble_size, ensemble_size, d_state)
        var_sum = var.unsqueeze(1) + var.unsqueeze(2)

        # shape: (n_actors, ensemble_size, ensemble_size, d_state)
        err = (mu_diff * 1 / var_sum * mu_diff)
        # shape: (n_actors, ensemble_size, ensemble_size)
        err = torch.sum(err, dim=-1)
        # shape: (n_actors, ensemble_size, ensemble_size)
        det = torch.sum(torch.log(var_sum), dim=-1)

        # shape: (n_actors, ensemble_size, ensemble_size)
        log_z = -0.5 * (err + det)
        # shape: (n_actors, ensemble_size * ensemble_size)
        log_z = log_z.reshape(n_act, es * es)
        # shape: (n_actors, 1)
        mx, _ = log_z.max(dim=1, keepdim=True)
        # shape: (n_actors, ensemble_size * ensemble_size)
        log_z = log_z - mx
        # shape: (n_actors, 1)
        exp = torch.exp(log_z).mean(dim=1, keepdim=True)
        # shape: (n_actors, 1)
        entropy_mean = -mx - torch.log(exp)
        # shape: (n_actors)
        entropy_mean = entropy_mean[:, 0]

        # mean of entropies
        # shape: (n_actors, ensemble_size)
        total_entropy = torch.sum(torch.log(var), dim=-1) #summing accross the latent dimension
        mean_entropy = total_entropy.mean(
            dim=1) / 2 + d_s * np.log(2) / 2    # shape: (n_actors) #average accross the ensemble size

        # jensen-renyi divergence
        # shape: (n_actors)
        utility = entropy_mean - mean_entropy

        return utility , mean_entropy #[24] , [24] batch size
    

    # ========== Jensen-Rényi Divergence ==========
# class JensenRenyiDivergence:
#     def __init__(self, states_mean, states_var):
#         self.states_mean = states_mean  # [E, B, D]
#         self.states_var = states_var

#     def compute_measure(self):
#         mean = self.states_mean  # [E, B, D]
#         var = self.states_var    # [E, B, D]
#         mean_mean = mean.mean(dim=0)  # [B, D]
#         var_mean = var.mean(dim=0)    # [B, D]
#         epistemic = ((mean - mean_mean) ** 2).mean(dim=0).mean(dim=-1)  # [B]
#         aleatoric = var_mean.mean(dim=-1)  # [B]
#         return epistemic, aleatoric
