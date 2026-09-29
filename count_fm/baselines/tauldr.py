"""Original tauLDR comparison implementation."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from count_fm.models import MLP

def make_tauldr_schedule(
    K,
    schedule = "exp",
    target_a_bar_1 = 1e-3,
    b = 10.0,
    beta0 = 0.1,
    beta1 = 20.0,
):
    if schedule == "exp":
        a = (-math.log(target_a_bar_1)) / (K * (b - 1.0))
        logb = math.log(b)

        def Lambda(t):
            bb = torch.tensor(b, device=t.device, dtype=t.dtype)
            return a * (torch.pow(bb, t) - 1.0)

        def beta(t):
            bb = torch.tensor(b, device=t.device, dtype=t.dtype)
            return a * torch.pow(bb, t) * logb

    elif schedule == "linear":
        def Lambda(t):
            return beta0 * t + 0.5 * (beta1 - beta0) * t * t

        def beta(t):
            return beta0 + (beta1 - beta0) * t
    else:
        raise ValueError(f"Unknown schedule={schedule}")

    def a_bar(t):
        return torch.exp(-K * Lambda(t))

    return beta, Lambda, a_bar


class TauLDRClassifier(nn.Module):
    def __init__(self, dim, K, w=64, C_max=None):
        super().__init__()
        self.dim = int(dim)
        self.K = int(K)
        self.C_max = (K - 1) if (C_max is None) else C_max
        self.net = MLP(dim=self.dim * self.K, out_dim=self.dim * self.K, w=w, time_varying=True)

    def forward(self, x, t):
        B, D = x.shape
        assert D == self.dim

        x_oh = F.one_hot(x.long(), num_classes=self.K).float().reshape(B, self.dim * self.K)

        if not torch.is_tensor(t):
            t = torch.full((B, 1), float(t), device=x.device)
        else:
            if t.ndim == 0:
                t = t.view(1, 1).expand(B, 1)
            elif t.ndim == 1:
                t = t.view(B, 1)
            t = t.to(device=x.device)

        x_cat = torch.cat([x_oh, t], dim=1)
        out = self.net(x_cat)
        return out.view(B, self.dim, self.K)


@torch.no_grad()
def tauldr_sample_xt(x0, t, C_max, a_bar_fn):
    K = C_max + 1
    B, D = x0.shape
    a_bar = a_bar_fn(t).view(B, 1)  # keep-prob
    keep = (torch.rand(B, D, device=x0.device) < a_bar)
    x_rand = torch.randint(0, K, (B, D), device=x0.device)
    return torch.where(keep, x0, x_rand)


@torch.no_grad()
def _sample_forward_jump_given_xt(xt, K):
    B, D = xt.shape
    d_idx = torch.randint(0, D, (B,), device=xt.device)
    cur = xt[torch.arange(B, device=xt.device), d_idx]  # [B]

    u = torch.randint(0, K - 1, (B,), device=xt.device)
    j = u + (u >= cur).long()  # skip cur

    tilde = xt.clone()
    tilde[torch.arange(B, device=xt.device), d_idx] = j
    return tilde, d_idx, cur


def tauldr_reverse_rates_all(classifier, x, t,
                             C_max, beta_fn,
                             a_bar_fn, eps=1e-12):
    K = C_max + 1
    B, D = x.shape

    logits = classifier(x, t)
    p0 = F.softmax(logits, dim=-1)

    a_bar = a_bar_fn(t).view(B, 1, 1)
    u = (1.0 - a_bar) / K

    onehot = F.one_hot(x.long(), num_classes=K).float()
    denom = (u + a_bar * onehot).clamp_min(eps)
    inv = 1.0 / denom

    # For uniform kernel, ratio-sum simplifies:
    # sum_k p0_k * q(j|k)/q(s|k) = u * sum_k p0_k * 1/denom_k + a_bar * p0_j * 1/denom_j
    weighted_inv_sum = (p0 * inv).sum(-1)
    base = (u.squeeze(-1).squeeze(-1)).view(B, 1) * weighted_inv_sum
    vec = base.unsqueeze(-1) + a_bar * (p0 * inv)

    beta = beta_fn(t).view(B, 1, 1)
    Rhat = beta * vec

    # remove self transitions
    Rhat = Rhat * (1.0 - onehot)

    lam_d = Rhat.sum(-1)
    Zhat = lam_d.sum(-1)
    return Rhat, Zhat, lam_d


def tauldr_train(
    train_counts,
    C_max,
    batch_size,
    num_epochs,
    classifier,
    opt,
    device=None,
    schedule="exp",
    t_min=1e-3,
    target_a_bar_1=1e-3,
    b=10.0,
    beta0=0.1,
    beta1=20.0,
):
    device = next(classifier.parameters()).device if device is None else device
    train_counts_t = torch.as_tensor(train_counts, dtype=torch.long, device=device)
    N, D = train_counts_t.shape
    K = C_max + 1

    beta_fn, _, a_bar_fn = make_tauldr_schedule(
        K=K, schedule=schedule,
        target_a_bar_1=target_a_bar_1, b=b,
        beta0=beta0, beta1=beta1,
    )

    steps_per_epoch = (N + batch_size - 1) // batch_size
    loss_hist = []

    for ep in tqdm(range(num_epochs)):
        for _ in range(steps_per_epoch):
            idx = torch.randint(0, N, (batch_size,), device=device)
            x0 = train_counts_t[idx]  # data

            t = torch.rand((batch_size, 1), device=device) * (1.0 - t_min) + t_min
            xt = tauldr_sample_xt(x0, t, C_max, a_bar_fn)

            tilde_xt, d_idx, x_old = _sample_forward_jump_given_xt(xt, K)

            # term 1: Zhat at xt
            _, Zhat, _ = tauldr_reverse_rates_all(classifier, xt, t, C_max, beta_fn, a_bar_fn)

            # term 2: log reverse rate from tilde_xt -> xt (only one dim differs)
            Rhat_tilde, _, _ = tauldr_reverse_rates_all(classifier, tilde_xt, t, C_max, beta_fn, a_bar_fn)
            R_back = Rhat_tilde[torch.arange(batch_size, device=device), d_idx, x_old]

            # Z_fwd(t) for uniform forward rates, factorized over dims:
            # each dim has exit rate beta(t)*(K-1)
            Z_fwd = (D * beta_fn(t).view(batch_size) * (K - 1))

            loss = (Zhat - Z_fwd * torch.log(R_back.clamp_min(1e-12))).mean()

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            loss_hist.append(loss.item())

    return classifier, loss_hist


@torch.no_grad()
def tauldr_sample(
    classifier,
    n_step,
    x_noise,
    C_max,
    device=None,
    schedule="exp",
    t_min=1e-3,
    target_a_bar_1=1e-3,
    b=10.0,
    beta0=0.1,
    beta1=20.0,
    return_traj=True,
):
    device = next(classifier.parameters()).device if device is None else device
    K = C_max + 1
    x = x_noise.clone().long().to(device)
    B, D = x.shape

    beta_fn, _, a_bar_fn = make_tauldr_schedule(
        K=K, schedule=schedule,
        target_a_bar_1=target_a_bar_1, b=b,
        beta0=beta0, beta1=beta1,
    )

    dt = (1.0 - t_min) / n_step
    traj = torch.zeros((n_step + 1, B, D), dtype=torch.long, device=device)
    traj[0] = x

    for s in range(n_step):
        t_cur = 1.0 - s * dt
        t = torch.full((B, 1), t_cur, device=device)

        Rhat, _, lam_d = tauldr_reverse_rates_all(classifier, x, t, C_max, beta_fn, a_bar_fn)

        # per-dim jump probability over dt
        p_jump = 1.0 - torch.exp(-lam_d * dt)
        jump = (torch.rand(B, D, device=device) < p_jump)

        probs = Rhat / (lam_d.unsqueeze(-1) + 1e-12)
        probs_flat = probs.view(B * D, K)
        sampled = torch.multinomial(probs_flat, num_samples=1).view(B, D)

        x = torch.where(jump, sampled, x)
        traj[s + 1] = x

    if return_traj:
        return x, traj
    return x
