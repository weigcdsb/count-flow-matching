"""Original discreteFM comparison implementation."""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

def kappa_and_deriv(t, schedule="quadratic"):
    if schedule == "linear":
        kappa = t
        kdot = torch.ones_like(t)
    elif schedule == "quadratic":
        kappa = t * t
        kdot = 2.0 * t
    elif schedule == "cosine":
        # κ(t) = 1 - cos(pi/2 * t)
        kappa = 1.0 - torch.cos(0.5 * math.pi * t)
        kdot = 0.5 * math.pi * torch.sin(0.5 * math.pi * t)
    else:
        raise ValueError(f"Unknown schedule={schedule}")
    return kappa.clamp(0.0, 1.0), kdot.clamp_min(1e-12)


class DiscreteFMClassifier(nn.Module):
    def __init__(self, num_dims, K, C_max, mlp_width=64):
        super().__init__()
        self.num_dims = int(num_dims)
        self.K = int(K)
        self.C_max = float(C_max)

        in_dim = self.num_dims * self.K + 1
        out_dim = self.num_dims * self.K

        self.net = nn.Sequential(
            nn.Linear(in_dim, mlp_width),
            nn.SiLU(),
            nn.Linear(mlp_width, mlp_width),
            nn.SiLU(),
            nn.Linear(mlp_width, out_dim),
        )

    def forward(self, x, t):
        B, D = x.shape
        assert D == self.num_dims

        x_oh = F.one_hot(x.long(), num_classes=self.K).float().reshape(B, self.num_dims * self.K)
        h = torch.cat([x_oh, t], dim=1)
        out = self.net(h)
        return out.view(B, self.num_dims, self.K)


@torch.no_grad()
def _sample_xt_from_coupling(x0, x1, kappa):
    B, D = x0.shape
    mask = (torch.rand(B, D, device=x0.device) < kappa)  # broadcast κ across D
    xt = torch.where(mask, x1, x0)
    return xt


def discrete_fm_train(
    train_counts,
    C_max,
    batch_size,
    num_epochs,
    classifier,
    opt,
    device,
    schedule="quadratic",
    t_min=1e-3,
    x0_mode="uniform",
    log_smooth=200,
):
    classifier.train()
    train_counts = train_counts.to(device)
    N, D = train_counts.shape
    K = int(C_max) + 1

    losses = []
    steps_per_epoch = (N + batch_size - 1) // batch_size
    pbar = tqdm(range(num_epochs), desc="[DiscreteFM]", leave=True)

    for ep in pbar:
        for _ in range(steps_per_epoch):
            idx = torch.randint(0, N, (batch_size,), device=device)
            x1 = train_counts[idx].long()
            B = x1.shape[0]

            if x0_mode == "uniform":
                x0 = torch.randint(0, K, (B, D), device=device).long()
            elif x0_mode == "dataset":
                idx0 = torch.randint(0, N, (batch_size,), device=device)
                x0 = train_counts[idx0].long()
            else:
                raise ValueError(f"Unknown x0_mode={x0_mode}")

            t = torch.rand(B, 1, device=device) * (1.0 - 2.0 * t_min) + t_min
            kappa, _ = kappa_and_deriv(t, schedule=schedule)
            xt = _sample_xt_from_coupling(x0, x1, kappa)

            logits = classifier(xt, t).clamp(-20.0, 20.0)
            loss = F.cross_entropy(
                logits.view(-1, K),
                x1.view(-1),
                reduction="mean",
            )

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            losses.append(float(loss.detach().cpu()))

        window = losses[-min(log_smooth, len(losses)):]
        avg_loss = float(np.mean(window))
        pbar.set_postfix(loss=avg_loss)

    return classifier, losses


@torch.no_grad()
def sample_discrete_fm(
    classifier,
    x0,                  # [B,D] long, starting noise sample
    C_max,
    num_steps,
    device,
    schedule="quadratic",
    t_min=1e-3,
    return_traj=True,
):
    """
    Euler sampling using probability velocity u from eq. (24) style:
      u_i(y, z) = κdot/(1-κ) * (p_theta(y | z,t) - 1[y=z])
    Sampling step:
      x_{t+h}^i ~ δ_{x_t^i} + h u_i(·, x_t)
    Implemented via:
      probs = scale * p_theta; then add (1-scale) mass to the current state.
    where scale = h * κdot/(1-κ), with an adaptive clamp to keep a valid PMF.
    """
    classifier.eval()
    x = x0.to(device).long()
    B, D = x.shape
    K = int(C_max) + 1

    traj = []
    if return_traj:
        traj.append(x.clone())

    h_base = 1.0 / float(num_steps)

    for s in tqdm(range(num_steps), desc="[DiscreteFM sample]", leave=False):
        # t in [t_min, 1 - t_min]
        t_scalar = t_min + (s / float(num_steps)) * (1.0 - 2.0 * t_min)
        t = torch.full((B, 1), float(t_scalar), device=device)

        kappa, kdot = kappa_and_deriv(t, schedule=schedule)

        # scale = h * κdot/(1-κ), but keep scale <= 1 (adaptive h)
        # h_adapt = min(h_base, (1-κ)/κdot)
        h_adapt = torch.minimum(
            torch.full_like(kappa, h_base),
            (1.0 - kappa) / kdot
        )
        scale = (h_adapt * kdot / (1.0 - kappa)).clamp(0.0, 1.0)  # [B,1]

        logits = classifier(x, t).clamp(-20.0, 20.0)  # [B,D,K]
        p = F.softmax(logits, dim=-1)                 # [B,D,K]

        probs = scale.view(B, 1, 1) * p               # sums to scale
        # add (1-scale) to the current state index
        probs.scatter_add_(
            2,
            x.unsqueeze(-1),
            (1.0 - scale).view(B, 1, 1).expand(B, D, 1),
        )

        # sample next x
        x_next = torch.multinomial(probs.view(B * D, K), 1).view(B, D)
        x = x_next

        if return_traj:
            traj.append(x.clone())

    if return_traj:
        traj = torch.stack(traj, dim=0)  # [T+1,B,D]
        return x, traj
    return x
