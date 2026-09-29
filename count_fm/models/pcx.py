"""Piriform-cortex conditional models and mean/Poisson baselines."""
import torch
from torch import nn
from .backbones import ChunkedAdaLNTransformer

class MeanMLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=128, depth=3):
        super().__init__()
        layers = []
        cur = in_dim
        for _ in range(depth):
            layers += [nn.Linear(cur, hidden), nn.SiLU()]
            cur = hidden
        layers += [nn.Linear(cur, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return torch.clamp(self.net(x), min=0.0)


class PoissonMLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=128, depth=3, log_cap=8.0, eps=1e-8):
        super().__init__()
        self.log_cap = log_cap
        self.eps = eps
        layers = []
        cur = in_dim
        for _ in range(depth):
            layers += [nn.Linear(cur, hidden), nn.SiLU()]
            cur = hidden
        layers += [nn.Linear(cur, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        log_rate = torch.clamp(self.net(x), max=self.log_cap)
        return torch.exp(log_rate) + self.eps


class CondCountFMNet(nn.Module):
    def __init__(
        self,
        d,
        cont_dim,
        cat_cardinalities,
        emb_dim=16,
        cond_dim=32,
        d_model=64,
        depth=4,
        n_heads=4,
        mlp_ratio=4.0,
        dropout=0.0,
        chunk_size=16,
        log_cap=12.0,
        eps_out=1e-6,
        eps_time=1e-4,
    ):
        super().__init__()
        self.d = d
        self.cont_dim = cont_dim
        self.cat_cardinalities = list(cat_cardinalities)
        self.n_cat = len(self.cat_cardinalities)
        self.eps_out = eps_out
        self.eps_time = eps_time
        self.log_cap = log_cap

        self.cat_nulls = self.cat_cardinalities[:]

        if self.cont_dim > 0:
            self.cont_proj = nn.Linear(self.cont_dim, emb_dim)
            self.cont_null = nn.Parameter(torch.zeros(emb_dim))
        else:
            self.cont_proj = None
            self.cont_null = None

        self.cat_embs = nn.ModuleList([nn.Embedding(K + 1, emb_dim) for K in self.cat_cardinalities])

        n_pieces = (1 if self.cont_dim > 0 else 0) + self.n_cat
        self.cond_fuse = nn.Sequential(
            nn.Linear(n_pieces * emb_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )

        self.dim_total = d + cond_dim
        self.backbone = ChunkedAdaLNTransformer(
            dim=self.dim_total,
            out_dim=self.dim_total,
            time_varying=True,
            d_model=d_model,
            depth=depth,
            n_heads=n_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            chunk_size=chunk_size,
        )

        self.out_head = nn.Sequential(
            nn.SiLU(),
            nn.Linear(self.dim_total, 2 * d),
        )

    def encode_cond(self, c_cont, c_cat, is_uncond_mask=None):
        pieces = []
        if self.cont_dim > 0:
            e_cont = self.cont_proj(c_cont)
            if is_uncond_mask is not None and is_uncond_mask.any():
                e_cont = e_cont.clone()
                e_cont[is_uncond_mask] = self.cont_null.view(1, -1)
            pieces.append(e_cont)
        for j, emb in enumerate(self.cat_embs):
            labels = c_cat[:, j]
            if is_uncond_mask is not None:
                labels = torch.where(is_uncond_mask, self.cat_nulls[j], labels)
            pieces.append(emb(labels))
        cond = torch.cat(pieces, dim=1)
        cond = self.cond_fuse(cond)
        return cond

    def forward(self, xt, t, c_cont, c_cat, is_uncond_mask=None):
        cond = self.encode_cond(c_cont, c_cat, is_uncond_mask=is_uncond_mask)
        z = torch.cat([torch.cat([xt, cond], dim=1), t], dim=1)
        h = self.backbone(z)
        out = self.out_head(h)

        lam_bar_raw, beta_bar_raw = out[:, :self.d], out[:, self.d:]
        lam_bar = torch.exp(torch.clamp(lam_bar_raw, max=self.log_cap)) + self.eps_out
        beta_bar = torch.exp(torch.clamp(beta_bar_raw, max=self.log_cap)) + self.eps_out

        denom = 1.0 - t + self.eps_time
        lam = lam_bar / denom
        beta = beta_bar / denom
        return lam, beta

    def forward_uncond(self, xt, t):
        B = xt.shape[0]
        c_cont = torch.zeros((B, self.cont_dim), device=xt.device, dtype=torch.float32)
        c_cat = torch.empty((B, self.n_cat), device=xt.device, dtype=torch.long)
        for j, null_idx in enumerate(self.cat_nulls):
            c_cat[:, j] = null_idx
        is_uncond = torch.ones((B,), device=xt.device, dtype=torch.bool)
        return self.forward(xt, t, c_cont, c_cat, is_uncond_mask=is_uncond)


class CondCountFMNet_MLP(nn.Module):
    def __init__(
        self,
        d,
        cont_dim,
        cat_cardinalities,
        emb_dim=16,
        hidden=128,
        depth=3,
        log_cap=12.0,
        eps_out=1e-6,
        eps_time=1e-4,
    ):
        super().__init__()
        self.d = d
        self.cont_dim = cont_dim
        self.cat_cardinalities = list(cat_cardinalities)
        self.n_cat = len(self.cat_cardinalities)
        self.eps_out = eps_out
        self.eps_time = eps_time
        self.log_cap = log_cap

        self.cat_nulls = self.cat_cardinalities[:]

        if self.cont_dim > 0:
            self.cont_proj = nn.Linear(self.cont_dim, emb_dim)
            self.cont_null = nn.Parameter(torch.zeros(emb_dim))
        else:
            self.cont_proj = None
            self.cont_null = None

        self.cat_embs = nn.ModuleList([nn.Embedding(K + 1, emb_dim) for K in self.cat_cardinalities])

        n_pieces = (1 if self.cont_dim > 0 else 0) + self.n_cat
        in_dim = d + 1 + n_pieces * emb_dim

        layers = []
        cur = in_dim
        for _ in range(depth):
            layers += [nn.Linear(cur, hidden), nn.SiLU()]
            cur = hidden
        layers += [nn.Linear(cur, 2 * d)]
        self.mlp = nn.Sequential(*layers)

    def encode_cond(self, c_cont, c_cat, is_uncond_mask=None):
        pieces = []
        if self.cont_dim > 0:
            e_cont = self.cont_proj(c_cont)
            if is_uncond_mask is not None and is_uncond_mask.any():
                e_cont = e_cont.clone()
                e_cont[is_uncond_mask] = self.cont_null.view(1, -1)
            pieces.append(e_cont)
        for j, emb in enumerate(self.cat_embs):
            labels = c_cat[:, j]
            if is_uncond_mask is not None:
                labels = torch.where(is_uncond_mask, self.cat_nulls[j], labels)
            pieces.append(emb(labels))
        return torch.cat(pieces, dim=1)

    def forward(self, xt, t, c_cont, c_cat, is_uncond_mask=None):
        cond = self.encode_cond(c_cont, c_cat, is_uncond_mask=is_uncond_mask)
        h = torch.cat([xt, t, cond], dim=1)
        out = self.mlp(h)

        lam_bar_raw, beta_bar_raw = out[:, :self.d], out[:, self.d:]
        lam_bar = torch.exp(torch.clamp(lam_bar_raw, max=self.log_cap)) + self.eps_out
        beta_bar = torch.exp(torch.clamp(beta_bar_raw, max=self.log_cap)) + self.eps_out

        denom = 1.0 - t + self.eps_time
        lam = lam_bar / denom
        beta = beta_bar / denom
        return lam, beta

    def forward_uncond(self, xt, t):
        B = xt.shape[0]
        c_cont = torch.zeros((B, self.cont_dim), device=xt.device, dtype=torch.float32)
        c_cat = torch.empty((B, self.n_cat), device=xt.device, dtype=torch.long)
        for j, null_idx in enumerate(self.cat_nulls):
            c_cat[:, j] = null_idx
        is_uncond = torch.ones((B,), device=xt.device, dtype=torch.bool)
        return self.forward(xt, t, c_cont, c_cat, is_uncond_mask=is_uncond)
