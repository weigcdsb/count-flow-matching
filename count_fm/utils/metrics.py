"""Count summary statistics and RBF MMD."""

import torch

def basic_stats(x):
    mean = x.mean(0)
    var = x.var(0, unbiased=False)
    zero_frac = (x == 0).float().mean(0)
    return mean, var, zero_frac


def corr_mat(x, eps=1e-8):
    x = x - x.mean(0, keepdim=True)
    cov = (x.T @ x) / x.shape[0]
    std = cov.diag().clamp_min(eps).sqrt()
    return cov / (std[:, None] * std[None, :])


def mmd_rbf(x, y, gamma=None):
    # x:[n,d], y:[m,d]
    if gamma is None:
        gamma = 1.0 / x.shape[1]
    def _kernel(a, b):
        # ||a-b||^2
        a2 = (a*a).sum(1, keepdim=True)
        b2 = (b*b).sum(1, keepdim=True)
        dist2 = a2 - 2.0 * (a @ b.T) + b2.T
        return torch.exp(-gamma * dist2)

    Kxx = _kernel(x, x)
    Kyy = _kernel(y, y)
    Kxy = _kernel(x, y)

    # use unbiased-ish version
    n = x.shape[0]
    m = y.shape[0]
    mmd2 = (Kxx.sum() - Kxx.diag().sum()) / (n*(n-1) + 1e-8) \
         + (Kyy.sum() - Kyy.diag().sum()) / (m*(m-1) + 1e-8) \
         - 2.0 * Kxy.mean()
    return mmd2.item()
