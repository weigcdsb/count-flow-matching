"""hc-3 conditional models, including the original auxiliary mean head."""
import torch
from torch import nn

class CondMLP(nn.Module):
    def __init__(self, in_dim, out_dim, width=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, width),
            nn.SELU(),
            nn.Linear(width, width),
            nn.SELU(),
            nn.Linear(width, width),
            nn.SELU(),
            nn.Linear(width, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class PoissonMLP(nn.Module):
    def __init__(self, in_dim, out_dim, width=128, log_cap=12.0, eps=1e-8):
        super().__init__()
        self.base = CondMLP(in_dim, out_dim, width=width)
        self.log_cap = log_cap
        self.eps = eps

    def forward(self, x):
        z = torch.clamp(self.base(x), max=self.log_cap)
        return torch.exp(z) + self.eps


class CountFMNetAux(nn.Module):
    def __init__(self, d, c_dim_cfg, width=128, log_cap=5.0, eps=1e-8):
        super().__init__()
        self.d = d
        self.c_dim_cfg = c_dim_cfg
        self.log_cap = log_cap
        self.eps = eps

        self.state_net = nn.Sequential(
            nn.Linear(d + 1, width),
            nn.SELU(),
            nn.Linear(width, width),
            nn.SELU(),
        )
        self.cond_net = nn.Sequential(
            nn.Linear(c_dim_cfg, width),
            nn.SELU(),
            nn.Linear(width, width),
            nn.SELU(),
        )
        self.fuse_net = nn.Sequential(
            nn.Linear(2 * width, width),
            nn.SELU(),
            nn.Linear(width, width),
            nn.SELU(),
        )

        self.lam_head = nn.Linear(width, d)
        self.beta_head = nn.Linear(width, d)
        self.mean_head = nn.Linear(width, d)

    def _positive(self, z):
        z = torch.clamp(z, max=self.log_cap)
        return torch.exp(z) + self.eps

    def forward(self, inp):
        x_state = inp[:, :self.d + 1]
        cond = inp[:, self.d + 1:]

        hs = self.state_net(x_state)
        hc = self.cond_net(cond)
        h = self.fuse_net(torch.cat([hs, hc], dim=1))

        lam = self._positive(self.lam_head(h))
        beta = self._positive(self.beta_head(h))
        mu_hat = self._positive(self.mean_head(hc))

        return torch.cat([lam, beta, mu_hat], dim=1)
