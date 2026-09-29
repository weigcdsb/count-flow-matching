"""Original poisson_jump comparison implementation."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

class PoissonJumpDecoder(nn.Module):
    def __init__(self, dim, C_max, hidden = 256, n_layers = 2):
        super().__init__()
        self.dim = int(dim)
        self.C_max = float(C_max)

        layers = []
        in_dim = self.dim + 1  # concat t
        h = int(hidden)

        layers += [nn.Linear(in_dim, h), nn.SiLU()]
        for _ in range(int(n_layers) - 1):
            layers += [nn.Linear(h, h), nn.SiLU()]
        layers += [nn.Linear(h, self.dim)]

        self.net = nn.Sequential(*layers)

    def forward(self, z, t):
        z_in = z.float() / self.C_max
        inp = torch.cat([z_in, t], dim=1)
        out = self.net(inp)
        xhat = F.softplus(out) + 1e-6
        return xhat


def _pjump_alpha(t, schedule="exp", alpha_end=1e-3):
    if schedule == "exp":
        beta = -math.log(float(alpha_end))
        return torch.exp(-beta * t)
    if schedule == "linear":
        return (1.0 - t) * (1.0 - float(alpha_end)) + float(alpha_end)
    if schedule == "cosine":
        a = torch.cos(0.5 * math.pi * t) ** 2
        return torch.clamp(a, min=float(alpha_end))
    raise ValueError(f"Unknown schedule={schedule}")


def pjump_train(
    train_counts,
    C_max,
    batch_size,
    num_epochs,
    decoder,
    opt,
    device,
    n_step=1000,
    schedule="exp",
    alpha_end=1e-3,
    lambda_scale=1.0,
    eps=1e-8,
):
    decoder.train()
    train_counts = train_counts.to(device)
    N, D = train_counts.shape

    losses = []
    steps_per_epoch = (N + batch_size - 1) // batch_size
    pbar = tqdm(range(num_epochs), desc="[PoissonJUMP]", leave=True)

    for ep in pbar:
        for _ in range(steps_per_epoch):
            idx = torch.randint(0, N, (batch_size,), device=device)
            x0 = train_counts[idx].long()
            B = x0.shape[0]

            t_idx = torch.randint(1, n_step + 1, (B, 1), device=device)
            t = t_idx.float() / float(n_step)

            alpha_t = _pjump_alpha(t, schedule=schedule, alpha_end=alpha_end)
            rate = float(lambda_scale) * alpha_t * x0.float()
            zt = torch.poisson(rate)

            xhat = decoder(zt, t)

            p = x0.float()
            q = xhat
            term1 = torch.where(
                p > 0,
                p * (torch.log(p + eps) - torch.log(q + eps)),
                torch.zeros_like(p),
            )
            loss = (term1 - (p - q)).mean()

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            losses.append(float(loss.detach().cpu()))

        avg_loss = sum(losses[-50:]) / len(losses[-50:])
        pbar.set_postfix(loss=float(avg_loss))

    return decoder, losses


@torch.no_grad()
def pjump_sample_counts(
    decoder,
    num_samples,
    D,
    C_max,
    n_step,
    device,
    schedule="exp",
    alpha_end=1e-3,
    lambda_scale=1.0,
    return_traj=True,
):
    decoder.eval()

    z = torch.zeros(num_samples, D, device=device)  # z_T

    def _to_counts(z_):
        x = torch.round(z_ / float(lambda_scale))
        x = torch.clamp(x, 0, int(C_max)).long()
        return x

    traj = []
    if return_traj:
        traj.append(_to_counts(z))

    # generation direction: step 0 is noise (z_T), step n_step is sample (z_0)
    for s in range(n_step):
        t_idx = n_step - s
        t = torch.full((num_samples, 1), float(t_idx) / float(n_step), device=device)

        alpha_t = _pjump_alpha(t, schedule=schedule, alpha_end=alpha_end)
        t_prev = torch.full((num_samples, 1), float(t_idx - 1) / float(n_step), device=device)
        alpha_prev = _pjump_alpha(t_prev, schedule=schedule, alpha_end=alpha_end)

        delta = (alpha_prev - alpha_t).clamp(min=0.0)
        xhat = decoder(z, t)
        inc = torch.poisson(float(lambda_scale) * delta * xhat)
        z = z + inc

        if return_traj:
            traj.append(_to_counts(z))

    x_final = _to_counts(z)
    if return_traj:
        traj = torch.stack(traj, dim=0)
        return x_final, traj
    return x_final
