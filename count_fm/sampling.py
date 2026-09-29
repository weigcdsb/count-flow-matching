"""First-order local unit-jump sampling."""

import torch
from .utils.inference import inference_mode

def jump_probabilities(birth, death, step_size):
    """Return stay/birth/death probabilities for nonnegative rates.

    Each coordinate makes at most one jump in a step. A zero total rate
    gives (1, 0, 0); expm1 preserves small jump probabilities.
    """
    total = birth + death
    jump = -torch.expm1(-total * step_size)
    denom = torch.where(total > 0, total, torch.ones_like(total))
    return 1.0 - jump, jump * (birth / denom), jump * (death / denom)


def sample_euler(nets, n_step, x0, device, eps_t = 1e-4, eps_log = 1e-8, separate_heads=False):
    with inference_mode(nets):
        if n_step <= 0 or not 0 < eps_t < 1:
            raise ValueError("Require n_step > 0 and 0 < eps_t < 1")
        xt = x0.to(device=device, dtype=torch.float32).clone()
        N, d = xt.shape
        t = torch.full((N, 1), 0.0, device=device, dtype=torch.float32)
        Delta = torch.tensor((1.0 - eps_t) / float(n_step),
                             device=device, dtype=torch.float32)
        traj = torch.empty(n_step + 1, N, d, device=device, dtype=torch.float32)
        traj[0] = xt

        for s in range(n_step):
            xt_t = torch.cat([xt, t], dim=1)
            if not separate_heads:
                out = nets(xt_t)  # [N, 2d]
                lambda_theta, beta_theta = out[:, :d], out[:, d:]
            else:
                lambda_theta = nets[0](xt_t)  # [N, d]
                beta_theta = nets[1](xt_t)  # [N, d]

            idx_0 = (xt <= 0).to(torch.float32)         # no death at 0
            mu_theta = (xt * beta_theta) * (1.0 - idx_0)
            p_none, p_birth, p_death = jump_probabilities(lambda_theta, mu_theta, Delta)
            probs3 = torch.stack([p_none, p_birth, p_death], dim=-1)  # [N,d,3]

            probs3_flat = probs3.reshape(-1, 3)
            choice = torch.multinomial(probs3_flat, 1).view(N, d)

            adj = (choice == 1).to(torch.float32) - (choice == 2).to(torch.float32)
            xt = torch.clamp(xt + adj, min=0.0)

            t = torch.minimum(t + Delta, torch.full_like(t, 1.0 - eps_t))
            traj[s + 1] = xt

        x1_samples = xt.to(torch.long)
        return x1_samples, traj
