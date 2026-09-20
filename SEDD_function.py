import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

# -----------------------------
# SEDD (Uniform) for counts
# -----------------------------
class SEDDUniformClassifier(nn.Module):
    def __init__(self, dim, K, C_max, hidden=64, depth=3):
        super().__init__()
        self.dim = int(dim)
        self.K = int(K)
        self.C_max = float(C_max)

        in_dim = self.dim * self.K + 1
        out_dim = self.dim * self.K

        layers = []
        w = int(hidden)
        layers += [nn.Linear(in_dim, w), nn.SiLU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(w, w), nn.SiLU()]
        layers += [nn.Linear(w, out_dim)]
        self.net = nn.Sequential(*layers)

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

        inp = torch.cat([x_oh, t], dim=1)
        out = self.net(inp)
        return out.view(B, self.dim, self.K)



# class SEDDUniformClassifier(nn.Module):
#     def __init__(self, dim, K, C_max, hidden=64, depth=3):
#         super().__init__()
#         self.dim = dim
#         self.K = K
#         self.C_max = float(C_max)

#         in_dim = 2 * dim
#         layers = []
#         w = hidden
#         layers.append(nn.Linear(in_dim, w))
#         layers.append(nn.SiLU())
#         for _ in range(depth - 1):
#             layers.append(nn.Linear(w, w))
#             layers.append(nn.SiLU())
#         layers.append(nn.Linear(w, dim * K))
#         self.net = nn.Sequential(*layers)

#     def forward(self, x, t):
#         if t.dim() == 1:
#             t = t[:, None]
#         x_in = x.float() / self.C_max
#         t_in = t.expand(-1, self.dim)
#         inp = torch.cat([x_in, t_in], dim=-1)
#         out = self.net(inp)
#         return out.view(-1, self.dim, self.K)

def sedd_sigma_and_deriv(t, schedule="loglinear",
                         sigma_min=1e-5, sigma_max=20.0,
                         eps=1e-3):
    if schedule == "geometric":
        # sigma(t) = sigma_min^(1-t) * sigma_max^t
        log_min = math.log(sigma_min)
        log_max = math.log(sigma_max)
        sigma = torch.exp((1.0 - t) * log_min + t * log_max)
        sigma_dot = sigma * (log_max - log_min)
        return sigma, sigma_dot

    if schedule == "loglinear":
        # sigma(t) = -log(1 - (1-eps)*t)
        a = (1.0 - eps)
        sigma = -torch.log1p(-a * t)
        sigma_dot = a / (1.0 - a * t)
        return sigma, sigma_dot

    if schedule == "linear":
        sigma = sigma_max * t
        sigma_dot = torch.full_like(t, float(sigma_max))
        return sigma, sigma_dot

    raise ValueError(f"Unknown schedule: {schedule}")


@torch.no_grad()
def _sedd_uniform_forward_sample_xt(x0, sigma, K):
    B, D = x0.shape
    sigma = sigma.view(B, 1)

    exp_term = torch.exp(-sigma)
    p_same = exp_term + (1.0 - exp_term) / float(K)
    p_other = (1.0 - exp_term) / float(K)

    u = torch.rand(B, D, device=x0.device)
    keep = (u < p_same)

    r = torch.randint(low=0, high=K - 1, size=(B, D), device=x0.device)
    new = r + (r >= x0).long()
    xt = torch.where(keep, x0, new).long()

    denom = torch.where(xt == x0, p_same.expand(B, D), p_other.expand(B, D))
    return xt, p_same.expand(B, D), p_other.expand(B, D), denom
    
# @torch.no_grad()
# def _sedd_uniform_forward_sample_xt(x0, sigma, K):
#     B, D = x0.shape
#     sigma = sigma.view(B, 1)

#     exp_term = torch.exp(-float(K) * sigma)
#     p_same = (1.0 / K) + (1.0 - 1.0 / K) * exp_term
#     p_other = (1.0 - exp_term) / K

#     u = torch.rand(B, D, device=x0.device)
#     keep = (u < p_same)

#     # sample uniformly among K-1 states != x0
#     r = torch.randint(low=0, high=K-1, size=(B, D), device=x0.device)
#     new = r + (r >= x0).long()
#     xt = torch.where(keep, x0, new).long()

#     # denom = p(xt | x0)
#     denom = torch.where(xt == x0, p_same.expand(B, D), p_other.expand(B, D))
#     return xt, p_same.expand(B, D), p_other.expand(B, D), denom


def sedd_train(
    train_counts,
    C_max,
    batch_size,
    num_epochs,
    classifier,
    opt,
    device,
    schedule="loglinear",
    sigma_min=1e-5,
    sigma_max=20.0,
    eps=1e-3,
    t_min=1e-4,
    log_every=200,
):
    classifier.train()
    train_counts = train_counts.to(device)
    N, D = train_counts.shape
    K = int(C_max) + 1

    losses = []
    states = torch.arange(K, device=device).view(1, 1, K)
    steps_per_epoch = (N + batch_size - 1) // batch_size

    pbar = tqdm(range(num_epochs), desc="[SEDD]", leave=True)
    for ep in pbar:
        for _ in range(steps_per_epoch):
            idx = torch.randint(0, N, (batch_size,), device=device)
            x0 = train_counts[idx].long()
            B = x0.shape[0]

            t = torch.rand(B, 1, device=device) * (1.0 - t_min) + t_min

            sigma, sigma_dot = sedd_sigma_and_deriv(
                t, schedule=schedule, sigma_min=sigma_min, sigma_max=sigma_max, eps=eps
            )

            xt, p_same, p_other, denom = _sedd_uniform_forward_sample_xt(x0, sigma, K)

            logits = classifier(xt, t)
            log_s = logits.clamp(-20.0, 20.0)
            s = torch.exp(log_s)

            base = (p_other / denom).unsqueeze(-1)
            ratio = base.expand(B, D, K).clone()
            ratio_x0 = (p_same / denom)
            ratio.scatter_(2, x0.unsqueeze(-1), ratio_x0.unsqueeze(-1))

            mask = (states != xt.unsqueeze(-1))

            term = (s - ratio * log_s) * mask.float()
            loss = (sigma_dot.view(B, 1, 1) * term).mean()

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            losses.append(float(loss.detach().cpu()))

        pbar.set_postfix(loss=float(loss.detach().cpu()))

    return classifier, losses


# def sedd_train(
#     train_counts,
#     C_max,
#     batch_size,
#     num_epochs,
#     classifier,
#     opt,
#     device,
#     schedule="loglinear",
#     sigma_min=1e-5,
#     sigma_max=20.0,
#     eps=1e-3,
#     t_min=1e-4,
#     log_every=200,
# ):
#     classifier.train()
#     N, D = train_counts.shape
#     K = int(C_max) + 1

#     losses = []
#     states = torch.arange(K, device=device).view(1, 1, K)

#     pbar = tqdm(range(num_epochs), desc="[SEDD]", leave=True)
#     for ep in pbar:
#         idx = torch.randint(0, N, (batch_size,), device=device)
#         x0 = train_counts[idx].long()
#         B = x0.shape[0]

#         # sample t ~ Uniform([t_min, 1])
#         t = torch.rand(B, 1, device=device) * (1.0 - t_min) + t_min

#         sigma, sigma_dot = sedd_sigma_and_deriv(
#             t, schedule=schedule, sigma_min=sigma_min, sigma_max=sigma_max, eps=eps
#         )

#         # forward sample xt and needed probs
#         xt, p_same, p_other, denom = _sedd_uniform_forward_sample_xt(x0, sigma, K)

#         # model outputs log s(x,t) for all y
#         logits = classifier(xt, t)
#         log_s = logits.clamp(-20.0, 20.0)
#         s = torch.exp(log_s)

#         # ratio r(y) = p(y|x0) / p(xt|x0)
#         # p(y|x0) is p_same if y==x0 else p_other
#         base = (p_other / denom).unsqueeze(-1)
#         ratio = base.expand(B, D, K).clone()
#         ratio_x0 = (p_same / denom)
#         ratio.scatter_(2, x0.unsqueeze(-1), ratio_x0.unsqueeze(-1))

#         # mask out y == xt
#         mask = (states != xt.unsqueeze(-1))

#         # DWDSE (constant K(ratio) dropped, does not affect gradients)
#         term = (s - ratio * log_s) * mask.float()
#         loss = (sigma_dot.view(B, 1, 1) * term).mean()

#         opt.zero_grad(set_to_none=True)
#         loss.backward()
#         opt.step()

#         losses.append(float(loss.detach().cpu()))
#         pbar.set_postfix(loss=float(loss.detach().cpu()))

#     return classifier, losses


@torch.no_grad()
def sample_sedd_uniform(
    classifier,
    num_samples,
    D,
    C_max,
    num_steps,
    device,
    schedule="loglinear",
    sigma_min=1e-5,
    sigma_max=20.0,
    eps=1e-3,
    t_min=0.0,
    x_init=None,
    return_traj=True,
):
    classifier.eval()
    K = int(C_max) + 1

    if x_init is None:
        x = torch.randint(low=0, high=K, size=(num_samples, D), device=device).long()
    else:
        x = x_init.to(device).long()
        assert x.shape == (num_samples, D)

    traj = [x.clone()] if return_traj else None

    t_grid = torch.linspace(1.0, float(t_min), steps=num_steps + 1, device=device)

    for j in range(num_steps):
        t_curr = t_grid[j].view(1, 1).expand(num_samples, 1)
        t_next = t_grid[j + 1].view(1, 1).expand(num_samples, 1)

        sigma_curr, _ = sedd_sigma_and_deriv(
            t_curr, schedule=schedule, sigma_min=sigma_min, sigma_max=sigma_max, eps=eps
        )
        sigma_next, _ = sedd_sigma_and_deriv(
            t_next, schedule=schedule, sigma_min=sigma_min, sigma_max=sigma_max, eps=eps
        )
        delta_sigma = (sigma_curr - sigma_next).clamp(min=0.0)

        logits = classifier(x, t_curr)
        log_s = logits.clamp(-20.0, 20.0)
        s = torch.exp(log_s)

        mass = delta_sigma.view(num_samples, 1, 1) * (s / float(K))
        mass.scatter_(2, x.unsqueeze(-1), 0.0)

        p_self = 1.0 - mass.sum(dim=-1, keepdim=True)
        p = mass.clone()
        p.scatter_(2, x.unsqueeze(-1), p_self)

        p = p.clamp(min=0.0)
        p = p / (p.sum(dim=-1, keepdim=True) + 1e-12)

        x = torch.multinomial(p.view(-1, K), num_samples=1).view(num_samples, D).long()

        if return_traj:
            traj.append(x.clone())

    if return_traj:
        traj = torch.stack(traj, dim=0)
        return x, traj
    return x





# @torch.no_grad()
# def sample_sedd_uniform(
#     classifier,
#     num_samples,
#     D,
#     C_max,
#     num_steps,
#     device,
#     schedule="loglinear",
#     sigma_min=1e-5,
#     sigma_max=20.0,
#     eps=1e-3,
#     t_min=0.0,
#     return_traj=True,
# ):
#     classifier.eval()
#     K = int(C_max) + 1

#     # base distribution is stationary of Q_uniform: uniform over states
#     x = torch.randint(low=0, high=K, size=(num_samples, D), device=device).long()

#     traj = [x.clone()] if return_traj else None

#     # time grid from 1 -> 0
#     t_grid = torch.linspace(1.0, float(t_min), steps=num_steps + 1, device=device)

#     for j in range(num_steps):
#         t_curr = t_grid[j].view(1, 1).expand(num_samples, 1)
#         t_next = t_grid[j + 1].view(1, 1).expand(num_samples, 1)

#         sigma_curr, _ = sedd_sigma_and_deriv(
#             t_curr, schedule=schedule, sigma_min=sigma_min, sigma_max=sigma_max, eps=eps
#         )
#         sigma_next, _ = sedd_sigma_and_deriv(
#             t_next, schedule=schedule, sigma_min=sigma_min, sigma_max=sigma_max, eps=eps
#         )
#         delta_sigma = (sigma_curr - sigma_next).clamp(min=0.0)

#         logits = classifier(x, t_curr)
#         log_s = logits.clamp(-20.0, 20.0)
#         s = torch.exp(log_s)

#         # zero out the "stay" component at current state
#         s_masked = s.clone()
#         s_masked.scatter_(2, x.unsqueeze(-1), 0.0)

#         # Euler tau-leaping update
#         mass = delta_sigma.view(num_samples, 1, 1) * s_masked
#         p_self = 1.0 - mass.sum(dim=-1, keepdim=True)

#         p = mass
#         p.scatter_(2, x.unsqueeze(-1), p_self)
        
#         p = p.clamp(min=0.0)
#         p = p / (p.sum(dim=-1, keepdim=True) + 1e-12)

#         # sample token-wise
#         x_new = torch.multinomial(p.view(-1, K), num_samples=1).view(num_samples, D)
#         x = x_new.long()

#         if return_traj:
#             traj.append(x.clone())

#     if return_traj:
#         traj = torch.stack(traj, dim=0)
#         return x, traj
#     return x
