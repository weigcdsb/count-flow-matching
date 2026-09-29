"""Signed-binomial bridge and rate-matching objective."""

import torch

def sample_xt(x0, x1, t):
    """Sample the signed-binomial bridge; t has shape [batch, 1]."""
    with torch.no_grad():
        diff = x1 - x0                     # [B,d]
        sign = torch.sign(diff)
        n    = diff.abs()                  # [B,d], integer counts
        t_full = t.expand_as(n)
        b = torch.binomial(n.float(), t_full)   # [B,d]
        return x0 + sign * b.to(x0.dtype)


def sample_rt(xt, x1, t, eps_t=1e-4):
    """Conditional [birth, death] rates for 0 <= t <= 1 - eps_t.

    The denominator floor only protects the truncated endpoint numerically;
    unlike adding eps_t, it preserves the bridge rates inside this interval.
    """
    with torch.no_grad():
        idx_b = (x1 > xt).to(torch.float32) # birth id
        idx_d = (xt > x1).to(torch.float32) # death id
        idx_0 = (xt <= 0).to(torch.float32) # boundary, no death

        denom = (1.0 - t).clamp_min(eps_t)
        lambda_star = idx_b * (x1 - xt) / denom
        mu_star = idx_d * (xt - x1) / denom * (1 - idx_0)
        
        # concatenate: [\lambda_{1:d}, \mu_{1:d}]
        rates_star = torch.cat([lambda_star, mu_star], dim=1)
        
        return rates_star, idx_0


def model_forward(xt_t, separate_heads, nets, d, idx_0):
    if not separate_heads:
        out = nets(xt_t)
        lambda_theta, beta_theta = out[:, :d], out[:, d:]
    else:
        net_b = nets[0]
        net_d = nets[1]
        lambda_theta = net_b(xt_t)
        beta_theta = net_d(xt_t)
        
    mu_theta = (xt_t[:, :d] * beta_theta) * (1.0 - idx_0)
    rates_theta = torch.cat([lambda_theta, mu_theta], dim=1)
    return rates_theta


def model_loss(loss_mode, rates_theta, rates_star, eps_log = 1e-8):
    if loss_mode == "l2":
        loss = ((rates_theta - rates_star)**2).sum(dim=1).mean()
    elif loss_mode == "poisson":
        u = rates_star
        v = rates_theta
        loss = (v - u*torch.log(v + eps_log)).sum(1).mean()
    else:
        raise ValueError("loss_mode must be 'poisson' or 'l2'")
    
    return loss
