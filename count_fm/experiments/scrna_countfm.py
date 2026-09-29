"""Unconditional scRNA count-FM adapter and checkpoint I/O."""
import os
import numpy as np
import torch
import count_fm as countfm
import count_fm.models as models
from .base import UncondModel
from count_fm.utils.inference import inference_mode
from count_fm.sampling import jump_probabilities

class CountFMModel(UncondModel):
    name = "count-FM"

    def __init__(
        self,
        device="cuda",
        num_epochs=500,
        batch_size=32,
        ot_pair_batch_size=None,   # NEW
        eps_t=1e-4,
        eps_log=1e-8,
        loss_mode="poisson",
        margin=2,
        C_max=None,
        # match 3_scRNA.ipynb behavior
        x0_mode_train="poisson",
        lam0_train=1.0,
        # OT coupling option
        ot_cost="none",
        # transformer backbone
        d_model=256,
        depth=8,
        n_heads=8,
        mlp_ratio=4.0,
        dropout=0.0,
        chunk_size=128,
        log_cap=50.0,
    ):
        self.device = torch.device(device if (device == "cpu" or torch.cuda.is_available()) else "cpu")

        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.ot_pair_batch_size = batch_size if ot_pair_batch_size is None else ot_pair_batch_size  # NEW
        self.eps_t = eps_t
        self.eps_log = eps_log
        self.loss_mode = loss_mode
        self.margin = margin
        self.C_max = C_max

        self.x0_mode_train = x0_mode_train
        self.lam0_train = lam0_train
        self.ot_cost = ot_cost

        self.d_model = d_model
        self.depth = depth
        self.n_heads = n_heads
        self.mlp_ratio = mlp_ratio
        self.dropout = dropout
        self.chunk_size = chunk_size
        self.log_cap = log_cap

        self.net = None
        self.d = None
        self.train_loss = None

        self.lr = 1e-4
        self.weight_decay = 0.0

    def fit(self, X_train_counts, lr=None, weight_decay=None, num_epochs=None, resume=True):
        lr = float(self.lr if lr is None else lr)
        weight_decay = float(self.weight_decay if weight_decay is None else weight_decay)
        num_epochs = int(self.num_epochs if num_epochs is None else num_epochs)

        X1_torch = torch.tensor(X_train_counts, dtype=torch.float32)  # CPU
        d = X1_torch.shape[1]
        self.d = d
        if self.C_max is None:
            self.C_max = int(X1_torch.max().item() + self.margin)

        # build net if needed
        if (not resume) or (self.net is None):
            net_base = models.ChunkedAdaLNTransformer(
                dim=d, out_dim=2 * d, time_varying=True,
                d_model=self.d_model, depth=self.depth, n_heads=self.n_heads,
                mlp_ratio=self.mlp_ratio, dropout=self.dropout, chunk_size=self.chunk_size,
            ).to(self.device)
            self.net = models.ChunkedAdaLNTransformer_rate(
                net_base, log_cap=self.log_cap, eps=self.eps_log
            ).to(self.device)

            self._opt = None  # reset optimizer if we rebuilt net

        # reuse optimizer state if possible
        if (resume and hasattr(self, "_opt") and self._opt is not None):
            opt = self._opt
            for g in opt.param_groups:
                g["lr"] = lr
                g["weight_decay"] = weight_decay
        else:
            opt = torch.optim.AdamW(self.net.parameters(), lr=lr, weight_decay=weight_decay)
            self._opt = opt

        self.net.train()

        out = countfm.CountFM_train(
            X1_torch=X1_torch,
            nets=self.net,
            optimizer=opt,
            num_epochs=num_epochs,
            batch_size=self.batch_size,
            device=self.device,
            separate_heads=False,
            X0_torch=None,
            x0_mode=self.x0_mode_train,
            ot_cost=self.ot_cost,
            ot_pair_batch_size=self.ot_pair_batch_size,   # NEW
            C_max=self.C_max,
            margin=self.margin,
            eps_t=self.eps_t,
            eps_log=self.eps_log,
            loss_mode=self.loss_mode,
            lam0_train=self.lam0_train,                   # NEW
        )

        if isinstance(out, tuple):
            self.net, self.train_loss = out
        else:
            self.net, self.train_loss = out, None

        print(
            f"count-FM: resume={resume} lr={lr} wd={weight_decay} "
            f"epochs={num_epochs} batch={self.batch_size} ot_pair_batch={self.ot_pair_batch_size}"
        )
        return self

    @torch.no_grad()
    def sample(self, n_samples, n_step=500, batch_eval=32, x0_mode=None, lam0=None):
        import torch
        assert self.net is not None
        assert self.d is not None
        d = int(self.d)
        x0_mode = self.x0_mode_train if x0_mode is None else x0_mode
        if lam0 is None:
            lam0 = self.lam0_train if x0_mode == "poisson" else self.C_max

        if x0_mode == "poisson":
            x0 = torch.poisson(torch.full((n_samples, d), float(lam0), dtype=torch.float32)).to(torch.int64)
        elif x0_mode == "uniform":
            C_max_eff = int(lam0)
            x0 = torch.randint(0, C_max_eff + 1, (n_samples, d), dtype=torch.int64)
        else:
            raise ValueError(f"Unknown x0_mode: {x0_mode}")

        x1, _ = sample_euler_batched(
            net=self.net,
            n_step=n_step,
            x0_cpu=x0,
            device=self.device,
            batch_eval=batch_eval,
            eps_t=self.eps_t,
            eps_log=self.eps_log,
        )
        return x1.numpy().astype(np.int64)


@torch.no_grad()
def sample_euler_batched(net, n_step, x0_cpu, device, batch_eval=32, eps_t=1e-4, eps_log=1e-8):
    with inference_mode(net):
        if n_step <= 0 or not 0 < eps_t < 1:
            raise ValueError("Require n_step > 0 and 0 < eps_t < 1")
        N, d = x0_cpu.shape
        outs = []
        Delta = (1.0 - eps_t) / float(n_step)
        Delta_t = torch.tensor(Delta, device=device, dtype=torch.float32)

        for start in range(0, N, batch_eval):
            end = min(start + batch_eval, N)
            xt = x0_cpu[start:end].to(device=device, dtype=torch.float32).clone()
            B = xt.shape[0]
            t = torch.full((B, 1), 0.0, device=device, dtype=torch.float32)

            for _ in range(n_step):
                xt_t = torch.cat([xt, t], dim=1)
                out = net(xt_t)
                lambda_theta, beta_theta = out[:, :d], out[:, d:]

                idx_0 = (xt <= 0).to(torch.float32)
                mu_theta = (xt * beta_theta) * (1.0 - idx_0)
                p_none, p_birth, p_death = jump_probabilities(lambda_theta, mu_theta, Delta_t)

                probs3 = torch.stack([p_none, p_birth, p_death], dim=-1)
                choice = torch.multinomial(probs3.reshape(-1, 3), 1).view(B, d)
                adj = (choice == 1).to(torch.float32) - (choice == 2).to(torch.float32)
                xt = torch.clamp(xt + adj, min=0.0)

                t = torch.minimum(t + Delta_t, torch.full_like(t, 1.0 - eps_t))

            outs.append(xt.to(torch.long).cpu())

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return torch.cat(outs, dim=0), None
def save_countfm(model, path_dir):
    os.makedirs(path_dir, exist_ok=True)

    payload = {
        "state_dict": model.net.state_dict(),
        "config": {
            "d": model.d,
            "eps_t": model.eps_t,
            "eps_log": model.eps_log,
            "loss_mode": model.loss_mode,
            "margin": model.margin,
            "C_max": model.C_max,
            "x0_mode_train": model.x0_mode_train,
            "lam0_train": model.lam0_train,
            "ot_cost": model.ot_cost,
            "batch_size": model.batch_size,
            "ot_pair_batch_size": model.ot_pair_batch_size,
            "num_epochs": model.num_epochs,
            "d_model": model.d_model,
            "depth": model.depth,
            "n_heads": model.n_heads,
            "mlp_ratio": model.mlp_ratio,
            "dropout": model.dropout,
            "chunk_size": model.chunk_size,
            "log_cap": model.log_cap,
        }
    }

    torch.save(payload, os.path.join(path_dir, "countfm.pt"))
    print("Saved count-FM to", path_dir)


def load_countfm(path_dir, device="cuda"):
    ckpt = torch.load(os.path.join(path_dir, "countfm.pt"), map_location="cpu")
    cfg = ckpt["config"]
    d = int(cfg["d"])

    dev = torch.device(device if (device == "cpu" or torch.cuda.is_available()) else "cpu")

    net_base = models.ChunkedAdaLNTransformer(
        dim=d,
        out_dim=2 * d,
        time_varying=True,
        d_model=int(cfg["d_model"]),
        depth=int(cfg["depth"]),
        n_heads=int(cfg["n_heads"]),
        mlp_ratio=float(cfg["mlp_ratio"]),
        dropout=float(cfg["dropout"]),
        chunk_size=int(cfg["chunk_size"]),
    ).to(dev)

    net = models.ChunkedAdaLNTransformer_rate(
        net_base,
        log_cap=float(cfg["log_cap"]),
        eps=float(cfg["eps_log"]),
    ).to(dev)

    net.load_state_dict(ckpt["state_dict"])

    m = CountFMModel(
        device=device,
        num_epochs=int(cfg.get("num_epochs", 500)),
        batch_size=int(cfg.get("batch_size", 32)),
        ot_pair_batch_size=int(cfg.get("ot_pair_batch_size", cfg.get("batch_size", 32))),
        eps_t=float(cfg["eps_t"]),
        eps_log=float(cfg["eps_log"]),
        loss_mode=str(cfg["loss_mode"]),
        margin=float(cfg["margin"]),
        C_max=cfg["C_max"],
        x0_mode_train=str(cfg["x0_mode_train"]),
        lam0_train=float(cfg.get("lam0_train", 1.0)),
        ot_cost=str(cfg["ot_cost"]),
        d_model=int(cfg["d_model"]),
        depth=int(cfg["depth"]),
        n_heads=int(cfg["n_heads"]),
        mlp_ratio=float(cfg["mlp_ratio"]),
        dropout=float(cfg["dropout"]),
        chunk_size=int(cfg["chunk_size"]),
        log_cap=float(cfg["log_cap"]),
    )
    m.net = net
    m.d = d
    print("Loaded count-FM from", path_dir)
    return m


def init_or_load_countfm_ot(path_ot, path_base, device="cuda", ot_cost="sym_poisson"):
    ot_ckpt = os.path.join(path_ot, "countfm.pt")

    if os.path.exists(ot_ckpt):
        m = load_countfm(path_ot, device=device)
        m.ot_cost = ot_cost
        m._needs_ot_training = False
        print("Loaded count-FM-OT from", path_ot)
        return m

    m = load_countfm(path_base, device=device)
    m.ot_cost = ot_cost
    m._needs_ot_training = True
    print("Initialized count-FM-OT from non-OT checkpoint:", path_base)
    print("Run the OT continuation-training cell, then save to:", path_ot)
    return m
