"""Minibatch endpoint coupling and count-space OT costs."""

import numpy as np
import torch
import ot

def _get_ot_cost_matrix(x0, x1, ot_cost="none", ot_eps=1e-8):
    key = ot_cost.lower()

    if key == "none":
        return None
    if key == "l1":
        return (x0[:, None, :] - x1[None, :, :]).abs().sum(dim=2)
    if key == "l2":
        return torch.cdist(x0.float(), x1.float(), p=2) ** 2


    if key in {"sym_poisson", "symmetric_poisson", "symmetric-poisson", "poisson"}:
        # algebraically equivalent to the original symmetric Poisson cost,
        # but avoids building a huge [B0, B1, d] tensor
        A = x0.float()                     # [B0, d]
        B = x1.float()                     # [B1, d]

        logA = torch.log(A + ot_eps)       # [B0, d]
        logB = torch.log(B + ot_eps)       # [B1, d]

        # sum_k [a_k log a_k]
        A_logA = (A * logA).sum(dim=1, keepdim=True)      # [B0, 1]
        # sum_k [b_k log b_k]
        B_logB = (B * logB).sum(dim=1).unsqueeze(0)       # [1, B1]

        # sum_k [a_k log b_k] and sum_k [b_k log a_k]
        cross_ab = A @ logB.T                              # [B0, B1]
        cross_ba = logA @ B.T                              # [B0, B1]

        M = A_logA + B_logB - cross_ab - cross_ba
        return M.clamp_min(0.0)

    raise ValueError(f"Unknown ot_cost: {ot_cost}")


def _soft_ot_pair_batch(x0, x1, ot_cost="none", ot_eps=1e-8, out_size=None):
    key = ot_cost.lower()

    B0 = x0.shape[0]
    B1 = x1.shape[0]

    if out_size is None:
        out_size = min(B0, B1)

    if key == "none":
        # keep behavior unchanged when out_size matches the input batch
        if out_size == B0 == B1:
            return x0, x1

        # if candidate batch is larger than optimizer batch, downsample back
        idx0 = torch.randint(0, B0, (out_size,), device=x0.device)
        idx1 = torch.randint(0, B1, (out_size,), device=x1.device)
        return x0[idx0], x1[idx1]

    M = _get_ot_cost_matrix(x0, x1, ot_cost=ot_cost, ot_eps=ot_eps)
    a = ot.unif(B0)
    b = ot.unif(B1)
    pi = ot.emd(a, b, M.detach().cpu().numpy())

    pi = np.asarray(pi, dtype=np.float64)
    if (not np.all(np.isfinite(pi))) or (pi.sum() <= 1e-12):
        p = np.ones(B0 * B1, dtype=np.float64) / float(B0 * B1)
    else:
        p = np.clip(pi, 0.0, None).reshape(-1)
        p = p / p.sum()

    choice = np.random.choice(B0 * B1, size=out_size, replace=True, p=p)
    i_np, j_np = np.divmod(choice, B1)

    i = torch.as_tensor(i_np, device=x0.device, dtype=torch.long)
    j = torch.as_tensor(j_np, device=x1.device, dtype=torch.long)

    return x0[i], x1[j]
