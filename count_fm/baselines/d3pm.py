"""Original D3PM comparison implementation."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from count_fm.models import MLP

def _alpha_bar_cosine(T, s = 0.008, eps = 1e-5, device=None):
    ts = torch.linspace(0, T, T + 1, device=device) / float(T)
    f = torch.cos((ts + s) / (1.0 + s) * math.pi / 2.0) ** 2
    f0 = f[0].clone()
    alpha_bar = (f / f0).clamp(min=eps, max=1.0)
    alpha_bar[0] = 1.0
    return alpha_bar


def _alpha_bar_linear_beta(T, beta_min = 1e-4, beta_max = 0.02, device=None):
    betas = torch.linspace(beta_min, beta_max, T, device=device)
    alphas = (1.0 - betas).clamp(min=1e-5, max=1.0)
    alpha_bar = torch.ones(T + 1, device=device)
    alpha_bar[1:] = torch.cumprod(alphas, dim=0)
    alpha_bar[0] = 1.0
    return alpha_bar


class D3PMClassifier(nn.Module):
    def __init__(self, num_dims, K, mlp_width = 64, time_varying= True):
        super().__init__()
        self.D = int(num_dims)
        self.K = int(K)
        self.mlp = MLP(dim=self.D * self.K,
                       out_dim=self.D * self.K,
                       w=mlp_width,
                       time_varying=time_varying)

    def forward(self, x_t, t_norm):
        B, D = x_t.shape
        assert D == self.D
        x_onehot = F.one_hot(x_t, num_classes=self.K).float()
        x_flat = x_onehot.reshape(B, D * self.K)

        if not torch.is_tensor(t_norm):
            t_norm = torch.full((B, 1), float(t_norm), device=x_t.device)
        else:
            if t_norm.ndim == 0:
                t_norm = t_norm.view(1, 1).expand(B, 1)
            elif t_norm.ndim == 1:
                t_norm = t_norm.view(B, 1)
            t_norm = t_norm.to(device=x_t.device)

        if getattr(self.mlp, "time_varying", False):
            x_in = torch.cat([x_flat, t_norm], dim=1)
        else:
            x_in = x_flat

        logits_flat = self.mlp(x_in)
        return logits_flat.view(B, D, self.K)


@torch.no_grad()
def _q_sample_uniform(x0, t_int, K, alpha_bar):
    
    B, D = x0.shape
    a_bar_t = alpha_bar[t_int].view(B, 1)
    keep = (torch.rand(B, D, device=x0.device) < a_bar_t)
    uni  = torch.randint(0, K, (B, D), device=x0.device)
    x_t  = torch.where(keep, x0, uni)
    return x_t


def _reverse_p_prev_from_x0pred_uniform(
    p_x0,
    x_t,
    t_int,
    K,
    alpha_bar,
    eps,
):
    B, D, _ = p_x0.shape

    a_bar_t   = alpha_bar[t_int].view(B, 1, 1)
    a_bar_prev = alpha_bar[(t_int - 1).clamp_min(0)].view(B, 1, 1)
    a_t = (alpha_bar[t_int] / alpha_bar[(t_int - 1).clamp_min(0)].clamp_min(1e-8)).view(B, 1, 1)
    a_t = torch.where(t_int.view(B, 1, 1) == 0, torch.ones_like(a_t), a_t)

    c_bar_t    = (1.0 - a_bar_t) / float(K)
    c_t        = (1.0 - a_t) / float(K)

    onehot_xt = F.one_hot(x_t, num_classes=K).float()

    # denom_vec[k] = q(x_t | x0=k) = c_bar_t + a_bar_t * 1[k == x_t]
    denom_vec = c_bar_t + a_bar_t * onehot_xt

    w = p_x0 / (denom_vec + eps)
    w_mean = w.mean(dim=-1, keepdim=True)

    # s = w @ Q_bar_{t-1} = a_bar_prev*w + (1-a_bar_prev)*mean(w)*1
    s = a_bar_prev * w + (1.0 - a_bar_prev) * w_mean

    # col[i] = q(x_t | x_{t-1}=i) = c_t + a_t * 1[i == x_t]
    col = c_t + a_t * onehot_xt

    p_prev = col * s
    p_prev = p_prev / (p_prev.sum(dim=-1, keepdim=True) + eps)
    return p_prev


def _q_posterior_prev_uniform(
    x0,
    x_t,
    t_int,
    K,
    alpha_bar,
    eps = 1e-8,
):
    B, D = x0.shape

    a_bar_t    = alpha_bar[t_int].view(B, 1, 1)
    a_bar_prev = alpha_bar[(t_int - 1).clamp_min(0)].view(B, 1, 1)
    a_t = (alpha_bar[t_int] / alpha_bar[(t_int - 1).clamp_min(0)].clamp_min(1e-8)).view(B, 1, 1)
    a_t = torch.where(t_int.view(B, 1, 1) == 0, torch.ones_like(a_t), a_t)

    c_bar_t    = (1.0 - a_bar_t) / float(K)
    c_bar_prev = (1.0 - a_bar_prev) / float(K)
    c_t        = (1.0 - a_t) / float(K)

    onehot_x0 = F.one_hot(x0, num_classes=K).float()
    onehot_xt = F.one_hot(x_t, num_classes=K).float()

    # q(x_{t-1} | x0): row of Q_bar_{t-1}
    prev_probs = c_bar_prev + a_bar_prev * onehot_x0

    # q(x_t | x_{t-1}): column of Q_t evaluated at x_t
    col = c_t + a_t * onehot_xt

    unnorm = prev_probs * col

    # denom = q(x_t | x0) = c_bar_t + a_bar_t * 1[x_t == x0]
    denom = c_bar_t.squeeze(-1) + a_bar_t.squeeze(-1) * (x_t == x0).float()
    q_post = unnorm / (denom.unsqueeze(-1) + eps)
    q_post = q_post / (q_post.sum(dim=-1, keepdim=True) + eps)
    return q_post


def d3pm_loss_uniform(
    classifier,
    counts_batch,
    K,
    alpha_bar,
    lambda_aux = 1e-3,
    eps = 1e-8,
):
    B, D = counts_batch.shape
    T = alpha_bar.shape[0] - 1

    # sample t per-sample (like your Dirichlet FM sampling alpha per-sample)
    t_int = torch.randint(low=1, high=T + 1, size=(B,), device=counts_batch.device)
    t_norm = t_int.float() / float(T)

    # forward sample x_t ~ q(x_t | x0)
    x_t = _q_sample_uniform(counts_batch, t_int, K, alpha_bar)

    # predict p~(x0 | x_t)
    logits = classifier(x_t, t_norm)
    logits_flat = logits.reshape(B * D, K)
    x0_flat = counts_batch.reshape(B * D)

    ce_flat = F.cross_entropy(logits_flat, x0_flat, reduction="none").view(B, D)
    ce_per_sample = ce_flat.mean(dim=1)
    aux_loss = ce_per_sample.mean()

    # VB term: for t==1 use L0 = -log p(x0|x1), for t>1 use cross-entropy of q_post vs p_theta_prev
    p_x0 = torch.softmax(logits, dim=-1)

    vb_per_sample = torch.zeros(B, device=counts_batch.device)

    is_t1 = (t_int == 1)
    if is_t1.any():
        vb_per_sample[is_t1] = ce_per_sample[is_t1]

    if (~is_t1).any():
        idx = (~is_t1).nonzero(as_tuple=False).squeeze(-1)

        x0_sub = counts_batch[idx]
        xt_sub = x_t[idx]
        t_sub  = t_int[idx]
        px0_sub = p_x0[idx]

        q_post = _q_posterior_prev_uniform(x0_sub, xt_sub, t_sub, K, alpha_bar, eps=eps)
        p_prev = _reverse_p_prev_from_x0pred_uniform(px0_sub, xt_sub, t_sub, K, alpha_bar, eps=eps)

        vb_terms = -(q_post * torch.log(p_prev + eps)).sum(dim=-1)
        vb_per_sample[idx] = vb_terms.mean(dim=1)

    vb_loss = vb_per_sample.mean()
    total_loss = vb_loss + float(lambda_aux) * aux_loss
    return total_loss, vb_loss.detach(), aux_loss.detach()


def D3PM_train(
    train_counts,
    C_max,
    batch_size,
    num_epochs,
    classifier,
    optimizer,
    device,
    num_steps=1000,
    schedule="cosine",
    lambda_aux=1e-3,
    beta_min=1e-4,
    beta_max=0.02,
    s_cos=0.008,
):
    train_counts = train_counts.to(device)
    N, D = train_counts.shape
    K = int(C_max) + 1

    if schedule == "cosine":
        alpha_bar = _alpha_bar_cosine(num_steps, s=s_cos).to(device)
    elif schedule == "linear":
        alpha_bar = _alpha_bar_linear_beta(num_steps, beta_min=beta_min, beta_max=beta_max).to(device)
    else:
        raise ValueError("schedule must be 'cosine' or 'linear'")

    loss_hist = []
    vb_hist = []
    aux_hist = []
    steps_per_epoch = (N + batch_size - 1) // batch_size

    for epoch in tqdm(range(num_epochs)):
        for _ in range(steps_per_epoch):
            idx = torch.randint(0, N, (batch_size,), device=device)
            counts_batch = train_counts[idx]

            optimizer.zero_grad(set_to_none=True)
            loss, vb, aux = d3pm_loss_uniform(
                classifier=classifier,
                counts_batch=counts_batch,
                K=K,
                alpha_bar=alpha_bar,
                lambda_aux=lambda_aux,
            )
            loss.backward()
            optimizer.step()

            loss_hist.append(float(loss.item()))
            vb_hist.append(float(vb.item()))
            aux_hist.append(float(aux.item()))

    return classifier, {"loss": loss_hist, "vb": vb_hist, "aux": aux_hist}


@torch.no_grad()
def sample_d3pm_counts(
    classifier,
    num_samples,
    D,
    C_max,
    num_steps,
    device,
    schedule = "cosine",
    beta_min = 1e-4,
    beta_max = 0.02,
    s_cos = 0.008,
    xT = None,
    return_traj = True,
    eps = 1e-8,
):
    K = int(C_max) + 1

    if schedule == "cosine":
        alpha_bar = _alpha_bar_cosine(num_steps, s=s_cos).to(device)
    elif schedule == "linear":
        alpha_bar = _alpha_bar_linear_beta(num_steps, beta_min=beta_min, beta_max=beta_max).to(device)
    else:
        raise ValueError("schedule must be 'cosine' or 'linear'")

    if xT is None:
        x_t = torch.randint(0, K, (num_samples, D), device=device, dtype=torch.long)
    else:
        x_t = xT.to(device).long()
        assert x_t.shape == (num_samples, D)

    traj = [x_t.clone()] if return_traj else None

    for t in range(num_steps, 0, -1):
        t_int = torch.full((num_samples,), t, device=device, dtype=torch.long)
        t_norm = t_int.float() / float(num_steps)

        logits = classifier(x_t, t_norm)
        p_x0 = torch.softmax(logits, dim=-1)

        if t == 1:
            # sample x0 ~ p~(x0 | x1)
            x0 = torch.multinomial(p_x0.view(num_samples * D, K), 1).view(num_samples, D)
            x_t = x0
        else:
            p_prev = _reverse_p_prev_from_x0pred_uniform(
                p_x0=p_x0,
                x_t=x_t,
                t_int=t_int,
                K=K,
                alpha_bar=alpha_bar,
                eps=eps,
            )  # [B,D,K]
            x_prev = torch.multinomial(p_prev.view(num_samples * D, K), 1).view(num_samples, D)
            x_t = x_prev

        if return_traj:
            traj.append(x_t.clone())

    if return_traj:
        traj = torch.stack(traj[::-1], dim=0)

    return x_t, traj
