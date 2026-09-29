"""PCx conditional training and CFG sampling from the implementation notebook."""
import copy
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader
from count_fm.bridge import sample_xt, sample_rt, model_loss
from count_fm.utils.inference import inference_mode
from count_fm.sampling import jump_probabilities

def eval_one_epoch(net, loader, eps_t=1e-4,
                   eps_log=1e-8, loss_mode="poisson",
                   C_max=None, margin=2,
                   x0_mode="uniform", x0_poisson_rate=1.0):
    device = next(net.parameters()).device
    net.eval()
    losses = []
    with torch.no_grad():
        for x1, ccont, ccat in loader:
            x1 = x1.to(device)
            ccont = ccont.to(device)
            ccat = ccat.to(device)

            B, d = x1.shape
            if C_max is None:
                C_max_local = int(x1.max().item() + margin)
            else:
                C_max_local = C_max

            # 
            if x0_mode == "uniform":
                if C_max is None:
                    C_max_local = int(x1.max().item() + margin)
                else:
                    C_max_local = int(C_max)

                x0 = torch.randint(0, C_max_local + 1, size=x1.shape,
                                   device=device, dtype=torch.long).to(x1.dtype)
            
            elif x0_mode == "poisson":
                x0 = torch.poisson(
                    torch.full_like(x1, float(x0_poisson_rate),
                                    dtype=torch.float32)
                ).to(x1.dtype)
            
            else:
                raise ValueError(f"Unknown x0_mode: {x0_mode}")

            # time and bridge
            t = torch.rand(B, 1, device=device) * (1.0 - eps_t)

            # u = torch.rand(B, 1, device=device)
            # gamma_t = 2.0   # try 2.0 first, then 1.5 or 3.0
            # z = 1.0 - (1.0 - u) ** gamma_t   # pushes mass toward 1 when gamma_t > 1
            # t = eps_t + (1.0 - eps_t) * z

            # one_minus_z = (1.0 - z).clamp_min(1e-8)
            # w = gamma_t * one_minus_z ** (1.0 - 1.0 / gamma_t)  # shape [B,1]
            # w = (w / w.mean()).view(-1)
            
            xt = sample_xt(x0, x1, t)
            rates_star, idx_0 = sample_rt(xt, x1, t, eps_t)

            lam, beta = net(xt, t, ccont, ccat, is_uncond_mask=None)
            mu = (xt * beta) * (1.0 - idx_0)
            rates_theta = torch.cat([lam, mu], dim=1)
            
            loss = model_loss(loss_mode, rates_theta, rates_star, eps_log)
            
            # if loss_mode == "l2":
            #     per = ((rates_theta - rates_star) ** 2).sum(dim=1)  # [B]
            # else:
            #     u_star = rates_star
            #     v_th   = rates_theta
            #     per = (v_th - u_star * torch.log(v_th + eps_log)).sum(dim=1)  # [B]
            # loss = (w * per).mean()
            
            losses.append(loss.item())

    return float(np.mean(losses)) if len(losses) else np.nan


def _make_epoch_train_monitor_loaders(train_loader, monitor_fraction=0.1, seed=0):
    ds = train_loader.dataset
    tensors = ds.tensors
    N = len(ds)

    n_monitor = int(round(monitor_fraction * N))
    n_monitor = max(1, min(n_monitor, N - 1))

    g = torch.Generator()
    g.manual_seed(int(seed))
    perm = torch.randperm(N, generator=g)

    monitor_idx = perm[:n_monitor]
    train_idx = perm[n_monitor:]

    train_epoch_ds = TensorDataset(*(t[train_idx] for t in tensors))
    monitor_epoch_ds = TensorDataset(*(t[monitor_idx] for t in tensors))

    batch_size = train_loader.batch_size
    train_epoch_loader = DataLoader(
        train_epoch_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=train_loader.drop_last,
    )
    monitor_epoch_loader = DataLoader(
        monitor_epoch_ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )
    return train_epoch_loader, monitor_epoch_loader


def train_cond_countfm_cfg(
    net,
    opt,
    train_loader,
    val_loader=None,                 # if None, use rotating monitor split inside training
    num_epochs=50,
    x0_mode="uniform",
    x0_poisson_rate=1.0,
    C_max=None,
    margin=2,
    eps_t=1e-4,
    eps_log=1e-8,
    loss_mode="poisson",
    p_uncond=0.1,
    monitor_fraction=0.1,           # only used if val_loader is None
    eval_every=5,
    patience=30,
    min_delta=1e-4,
    scheduler_factor=0.5,
    scheduler_patience=8,
    min_lr=1e-5,
    split_seed=0,
):
    device = next(net.parameters()).device
    if C_max is None:
        all_x = train_loader.dataset.tensors[0]
        C_max = int(all_x.max().item() + margin)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt,
        mode="min",
        factor=scheduler_factor,
        patience=scheduler_patience,
        min_lr=min_lr,
    )

    best_metric = np.inf
    best_state = copy.deepcopy(net.state_dict())
    best_epoch = -1
    bad_count = 0
    history = []

    print(
        "C_max:", C_max,
        "p_uncond:", p_uncond,
        "x0_mode:", x0_mode,
        "monitor_fraction:", monitor_fraction if val_loader is None else "fixed_val_loader",
    )

    for ep in range(num_epochs):
        if val_loader is None:
            train_epoch_loader, monitor_loader = _make_epoch_train_monitor_loaders(
                train_loader,
                monitor_fraction=monitor_fraction,
                seed=split_seed + ep,
            )
        else:
            train_epoch_loader = train_loader
            monitor_loader = val_loader

        net.train()
        train_losses = []

        for x1, ccont, ccat in train_epoch_loader:
            x1 = x1.to(device)
            ccont = ccont.to(device)
            ccat = ccat.to(device)

            B, d = x1.shape

            if x0_mode == "uniform":
                x0 = torch.randint(
                    0, C_max + 1,
                    size=x1.shape,
                    device=device,
                    dtype=torch.long,
                ).to(x1.dtype)

            elif x0_mode == "poisson":
                x0 = torch.poisson(
                    torch.full_like(x1, float(x0_poisson_rate), dtype=torch.float32)
                ).to(x1.dtype)

            else:
                raise ValueError("x0_mode must be 'uniform' or 'poisson'")

            t = torch.rand(B, 1, device=device) * (1.0 - eps_t)
            xt = sample_xt(x0, x1, t)
            rates_star, idx_0 = sample_rt(xt, x1, t, eps_t)

            drop = (torch.rand(B, device=device) < p_uncond)
            if drop.any():
                ccont = ccont.clone()
                ccat = ccat.clone()
                for j, null_idx in enumerate(net.cat_nulls):
                    ccat[drop, j] = null_idx

            lam, beta = net(xt, t, ccont, ccat, is_uncond_mask=drop)
            mu = (xt * beta) * (1.0 - idx_0)
            rates_theta = torch.cat([lam, mu], dim=1)

            loss = model_loss(loss_mode, rates_theta, rates_star, eps_log)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()

            train_losses.append(loss.item())

        train_loss = float(np.mean(train_losses)) if len(train_losses) else np.nan

        if (ep % eval_every == 0) or (ep == num_epochs - 1):
            monitor_loss = eval_one_epoch(
                net,
                monitor_loader,
                eps_t=eps_t,
                eps_log=eps_log,
                loss_mode=loss_mode,
                C_max=C_max,
                x0_mode=x0_mode,
                x0_poisson_rate=x0_poisson_rate,
            )

            scheduler.step(monitor_loss)
            lr_now = opt.param_groups[0]["lr"]

            history.append({
                "epoch": ep,
                "train_loss": train_loss,
                "monitor_loss": monitor_loss,
                "lr": lr_now,
            })

            improved = monitor_loss < (best_metric - min_delta)
            if improved:
                best_metric = monitor_loss
                best_state = copy.deepcopy(net.state_dict())
                best_epoch = ep
                bad_count = 0
            else:
                bad_count += 1

            print(
                f"[ep {ep}] train_loss={train_loss:.6f}  "
                f"monitor_loss={monitor_loss:.6f}  lr={lr_now:.2e}  "
                f"best={best_metric:.6f} @ {best_epoch}"
            )

            if bad_count >= patience:
                print(f"Early stop at ep {ep}. Restore best epoch {best_epoch}.")
                break

    net.load_state_dict(best_state)
    return net, C_max, history
@torch.no_grad()
def sample_euler_cfg_condcountfm(
    net,
    n_step,
    x0,
    ccont,
    ccat,
    guidance_scale=1.0,
    eps_t=1e-4,
    eps_log=1e-8,
):
    with inference_mode(net):
        if n_step <= 0 or not 0 < eps_t < 1:
            raise ValueError("Require n_step > 0 and 0 < eps_t < 1")
        device = next(net.parameters()).device
        xt = x0.to(device=device, dtype=torch.float32).clone()
        N, d = xt.shape

        ccont = ccont.to(device=device, dtype=torch.float32)
        ccat  = ccat.to(device=device, dtype=torch.long)

        t = torch.full((N, 1), 0.0, device=device, dtype=torch.float32)
        Delta = torch.tensor((1.0 - eps_t) / float(n_step), device=device, dtype=torch.float32)

        ccont_u = torch.zeros((N, net.cont_dim), device=device, dtype=torch.float32)
        ccat_u = ccat.clone()
        for j, null_idx in enumerate(net.cat_nulls):
            ccat_u[:, j] = null_idx
        uncond_mask = torch.ones((N,), device=device, dtype=torch.bool)

        for _ in range(n_step):
            lam_c, beta_c = net(xt, t, ccont,  ccat,  is_uncond_mask=None)
            lam_u, beta_u = net(xt, t, ccont_u, ccat_u, is_uncond_mask=uncond_mask)

            lam  = lam_u  + guidance_scale * (lam_c  - lam_u)
            beta = beta_u + guidance_scale * (beta_c - beta_u)
            lam  = F.relu(lam)  + 1e-8
            beta = F.relu(beta) + 1e-8

            idx_0 = (xt <= 0).to(torch.float32)
            mu = (xt * beta) * (1.0 - idx_0)
            p_none, p_birth, p_death = jump_probabilities(lam, mu, Delta)

            probs3 = torch.stack([p_none, p_birth, p_death], dim=-1)
            probs3_flat = probs3.reshape(-1, 3)
            probs3_flat = probs3_flat / probs3_flat.sum(dim=1, keepdim=True)

            choice = torch.multinomial(probs3_flat, 1).view(N, d)
            adj = (choice == 1).to(torch.float32) - (choice == 2).to(torch.float32)
            xt = torch.clamp(xt + adj, min=0.0)

            t = torch.minimum(t + Delta, torch.full_like(t, 1.0 - eps_t))

        return xt.to(torch.long)
