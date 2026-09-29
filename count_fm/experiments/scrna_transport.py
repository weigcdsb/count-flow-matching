"""Developmental scRNA transport with the original cluster-aware OT recipe."""
import copy
import numpy as np
import torch
import ot
from count_fm.coupling import _get_ot_cost_matrix
from tqdm.auto import tqdm
from count_fm.bridge import sample_xt, sample_rt, model_forward, model_loss
from count_fm.models import ChunkedAdaLNTransformer, ChunkedAdaLNTransformer_rate
from count_fm.utils.inference import inference_mode
from count_fm.sampling import jump_probabilities



def _soft_ot_pair_batch(
    x0,
    x1,
    ot_cost="none",
    ot_eps=1e-8,
    src_clusters=None,
    tgt_clusters=None,
    allowed_targets=None,
    big_cost=1e12,
):
    key = ot_cost.lower()
    if key == "none":
        return x0, x1

    B0 = x0.shape[0]
    B1 = x1.shape[0]

    allowed = None
    M = _get_ot_cost_matrix(x0, x1, ot_cost=ot_cost, ot_eps=ot_eps)

    if (src_clusters is not None) and (tgt_clusters is not None) and (allowed_targets is not None):
        src_clusters = np.asarray(src_clusters).astype(str)
        tgt_clusters = np.asarray(tgt_clusters).astype(str)

        allowed = np.zeros((B0, B1), dtype=bool)
        for i, s in enumerate(src_clusters):
            allowed[i, :] = np.isin(tgt_clusters, list(allowed_targets[s]))

        if not allowed.any(axis=1).all():
            bad = np.where(~allowed.any(axis=1))[0]
            raise ValueError(f"Some source cells in this batch have no allowed targets: {src_clusters[bad].tolist()}")

        if not allowed.any(axis=0).all():
            bad = np.where(~allowed.any(axis=0))[0]
            raise ValueError(f"Some target cells in this batch have no allowed sources: {tgt_clusters[bad].tolist()}")

        allowed_t = torch.as_tensor(allowed, device=M.device, dtype=torch.bool)
        M = M.clone()
        finite_max = M[allowed_t].max() if allowed_t.any() else torch.tensor(0.0, device=M.device)
        M[~allowed_t] = finite_max + big_cost

    a = ot.unif(B0)
    b = ot.unif(B1)
    pi = ot.emd(a, b, M.detach().cpu().numpy())

    pi = np.asarray(pi, dtype=np.float64)
    if allowed is not None:
        if not np.all(np.isfinite(pi)) or pi.sum() <= 1e-12 or pi[~allowed].sum() > 1e-8:
            raise ValueError("Cluster constraints are infeasible for this balanced OT batch; adjust the batch composition or allowed_targets.")
        pi[~allowed] = 0.0
    if (not np.all(np.isfinite(pi))) or (pi.sum() <= 1e-12):
        p = np.ones(B0 * B1, dtype=np.float64) / float(B0 * B1)
    else:
        p = np.clip(pi, 0.0, None).reshape(-1)
        p = p / p.sum()

    choice = np.random.choice(B0 * B1, size=B1, replace=True, p=p)
    i_np, j_np = np.divmod(choice, B1)

    i = torch.as_tensor(i_np, device=x0.device, dtype=torch.long)
    j = torch.as_tensor(j_np, device=x1.device, dtype=torch.long)

    return x0[i], x1[j]


def _build_allowed_target_index_map(tgt_clusters, allowed_targets):
    tgt_clusters = np.asarray(tgt_clusters).astype(str)
    out = {}
    for src in sorted(allowed_targets):
        idx = np.where(np.isin(tgt_clusters, list(allowed_targets[src])))[0]
        out[src] = idx
    return out


def _sample_compatible_targets_for_sources(src_clusters_batch, allowed_target_index_map):
    src_clusters_batch = np.asarray(src_clusters_batch).astype(str)
    idx = np.empty(len(src_clusters_batch), dtype=np.int64)
    for i, s in enumerate(src_clusters_batch):
        pool = allowed_target_index_map[s]
        if len(pool) == 0:
            raise ValueError(f"No target cells available for source cluster {s}")
        idx[i] = np.random.choice(pool)
    return idx


@torch.no_grad()
def CountFM_eval(
    X1_torch,
    nets,
    batch_size,
    device,
    separate_heads=False,
    X0_torch=None,
    x0_mode="uniform",
    ot_cost="none",
    C_max=None,
    margin=2,
    eps_t=1e-4,
    eps_log=1e-8,
    loss_mode="poisson",
    X0_clusters=None,
    X1_clusters=None,
    allowed_targets=None,
    n_eval_batches=None,
):
    nets.eval()

    X1_torch = X1_torch.to(device)
    N1, d = X1_torch.shape

    if X0_torch is not None:
        X0_torch = X0_torch.to(device)
        N0 = X0_torch.shape[0]
    else:
        N0 = N1

    if C_max is None:
        C_max = int(X1_torch.max().item() + margin)

    if allowed_targets is not None:
        if X0_clusters is None or X1_clusters is None:
            raise ValueError("Need X0_clusters and X1_clusters when allowed_targets is used.")
        X0_clusters = np.asarray(X0_clusters).astype(str)
        X1_clusters = np.asarray(X1_clusters).astype(str)
        allowed_target_index_map = _build_allowed_target_index_map(X1_clusters, allowed_targets)
    else:
        allowed_target_index_map = None

    if n_eval_batches is None:
        n_eval_batches = max(1, (max(N0, N1) + batch_size - 1) // batch_size)

    losses = []

    for _ in range(n_eval_batches):
        if X0_torch is not None and x0_mode == "dataset":
            idx0 = torch.randint(0, N0, (batch_size,), device=device)
            x0 = X0_torch[idx0]
            src_clusters_batch = None if X0_clusters is None else np.asarray(X0_clusters)[idx0.detach().cpu().numpy()]
        elif x0_mode == "uniform":
            x0 = torch.randint(
                low=0,
                high=C_max + 1,
                size=(batch_size, d),
                device=device,
                dtype=X1_torch.dtype,
            )
            src_clusters_batch = None
        elif x0_mode == "poisson":
            x0 = torch.poisson(
                torch.full((batch_size, d), 1.0, device=device, dtype=torch.float32)
            ).to(X1_torch.dtype)
            src_clusters_batch = None
        else:
            raise ValueError(f"Unknown x0_mode: {x0_mode}")

        if (x0_mode == "dataset") and (allowed_target_index_map is not None):
            idx1_np = _sample_compatible_targets_for_sources(
                src_clusters_batch,
                allowed_target_index_map,
            )
            idx1 = torch.as_tensor(idx1_np, device=device, dtype=torch.long)
            x1 = X1_torch[idx1]
            tgt_clusters_batch = X1_clusters[idx1_np]
        else:
            idx1 = torch.randint(0, N1, (batch_size,), device=device)
            x1 = X1_torch[idx1]
            tgt_clusters_batch = None if X1_clusters is None else X1_clusters[idx1.detach().cpu().numpy()]

        x0, x1 = _soft_ot_pair_batch(
            x0,
            x1,
            ot_cost=ot_cost,
            ot_eps=eps_log,
            src_clusters=src_clusters_batch,
            tgt_clusters=tgt_clusters_batch,
            allowed_targets=allowed_targets,
        )

        B = x1.shape[0]
        t = torch.rand(B, 1, device=device) * (1.0 - eps_t)
        xt = sample_xt(x0, x1, t)
        rates_star, idx_0 = sample_rt(xt, x1, t, eps_t)
        xt_t = torch.cat([xt, t], dim=1)
        rates_theta = model_forward(xt_t, separate_heads, nets, d, idx_0)
        loss = model_loss(loss_mode, rates_theta, rates_star, eps_log)
        losses.append(float(loss))

    return float(np.mean(losses))


def CountFM_train(
    X1_torch,
    nets,
    optimizer,
    num_epochs,
    batch_size,
    device,
    separate_heads=False,
    X0_torch=None,
    x0_mode="uniform",
    ot_cost="none",
    C_max=None,
    margin=2,
    eps_t=1e-4,
    eps_log=1e-8,
    loss_mode="poisson",
    X0_clusters=None,
    X1_clusters=None,
    allowed_targets=None,
    X1_val_torch=None,
    X0_val_torch=None,
    X0_clusters_val=None,
    X1_clusters_val=None,
    eval_every=5,
    patience=30,
    min_delta=1e-4,
    scheduler_factor=0.5,
    scheduler_patience=8,
    min_lr=1e-5,
    grad_clip=1.0,
    n_eval_batches=40,
):
    X1_torch = X1_torch.to(device)
    N1, d = X1_torch.shape

    if X0_torch is not None:
        X0_torch = X0_torch.to(device)
        N0 = X0_torch.shape[0]
    else:
        N0 = N1

    if C_max is None:
        C_max = int(X1_torch.max().item() + margin)

    if allowed_targets is not None:
        if X0_clusters is None or X1_clusters is None:
            raise ValueError("Need X0_clusters and X1_clusters when allowed_targets is used.")
        X0_clusters = np.asarray(X0_clusters).astype(str)
        X1_clusters = np.asarray(X1_clusters).astype(str)
        if len(X0_clusters) != N0:
            raise ValueError("len(X0_clusters) must equal len(X0_torch)")
        if len(X1_clusters) != N1:
            raise ValueError("len(X1_clusters) must equal len(X1_torch)")
        allowed_target_index_map = _build_allowed_target_index_map(X1_clusters, allowed_targets)
    else:
        allowed_target_index_map = None

    use_val = (X1_val_torch is not None)
    if use_val and allowed_targets is not None:
        X0_clusters_val = np.asarray(X0_clusters_val).astype(str)
        X1_clusters_val = np.asarray(X1_clusters_val).astype(str)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=scheduler_factor,
        patience=scheduler_patience,
        min_lr=min_lr,
    )

    best_metric = np.inf
    best_state = copy.deepcopy(nets.state_dict())
    best_epoch = -1
    bad_count = 0
    history = []

    steps_per_epoch = (max(N0, N1) + batch_size - 1) // batch_size

    for epoch in tqdm(range(num_epochs)):
        nets.train()

        if X0_torch is not None and x0_mode == "dataset":
            perm0 = torch.randperm(N0, device=device)
        else:
            perm0 = None

        train_losses = []

        for step in range(steps_per_epoch):
            start = step * batch_size
            base = torch.arange(batch_size, device=device)

            if X0_torch is not None and x0_mode == "dataset":
                idx0 = perm0[(start + base) % N0]
                x0 = X0_torch[idx0]
                src_clusters_batch = None if X0_clusters is None else np.asarray(X0_clusters)[idx0.detach().cpu().numpy()]
            elif x0_mode == "uniform":
                x0 = torch.randint(
                    low=0,
                    high=C_max + 1,
                    size=(batch_size, d),
                    device=device,
                    dtype=X1_torch.dtype,
                )
                src_clusters_batch = None
            elif x0_mode == "poisson":
                x0 = torch.poisson(
                    torch.full((batch_size, d), 1.0, device=device, dtype=torch.float32)
                ).to(X1_torch.dtype)
                src_clusters_batch = None
            else:
                raise ValueError(f"Unknown x0_mode: {x0_mode}")

            if (x0_mode == "dataset") and (allowed_target_index_map is not None):
                idx1_np = _sample_compatible_targets_for_sources(
                    src_clusters_batch,
                    allowed_target_index_map,
                )
                idx1 = torch.as_tensor(idx1_np, device=device, dtype=torch.long)
                x1 = X1_torch[idx1]
                tgt_clusters_batch = X1_clusters[idx1_np]
            else:
                idx1 = torch.randint(0, N1, (batch_size,), device=device)
                x1 = X1_torch[idx1]
                tgt_clusters_batch = None if X1_clusters is None else X1_clusters[idx1.detach().cpu().numpy()]

            x0, x1 = _soft_ot_pair_batch(
                x0,
                x1,
                ot_cost=ot_cost,
                ot_eps=eps_log,
                src_clusters=src_clusters_batch,
                tgt_clusters=tgt_clusters_batch,
                allowed_targets=allowed_targets,
            )

            B = x1.shape[0]
            t = torch.rand(B, 1, device=device) * (1.0 - eps_t)
            xt = sample_xt(x0, x1, t)
            rates_star, idx_0 = sample_rt(xt, x1, t, eps_t)
            xt_t = torch.cat([xt, t], dim=1)
            rates_theta = model_forward(xt_t, separate_heads, nets, d, idx_0)
            loss = model_loss(loss_mode, rates_theta, rates_star, eps_log)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(nets.parameters(), grad_clip)
            optimizer.step()

            train_losses.append(float(loss))

        train_loss = float(np.mean(train_losses))

        if (epoch % eval_every == 0) or (epoch == num_epochs - 1):
            if use_val:
                monitor_loss = CountFM_eval(
                    X1_torch=X1_val_torch,
                    nets=nets,
                    batch_size=batch_size,
                    device=device,
                    separate_heads=separate_heads,
                    X0_torch=X0_val_torch,
                    x0_mode=x0_mode,
                    ot_cost=ot_cost,
                    C_max=C_max,
                    margin=margin,
                    eps_t=eps_t,
                    eps_log=eps_log,
                    loss_mode=loss_mode,
                    X0_clusters=X0_clusters_val,
                    X1_clusters=X1_clusters_val,
                    allowed_targets=allowed_targets,
                    n_eval_batches=n_eval_batches,
                )
            else:
                monitor_loss = train_loss

            scheduler.step(monitor_loss)
            lr_now = optimizer.param_groups[0]["lr"]

            history.append({
                "epoch": epoch,
                "train_loss": train_loss,
                "monitor_loss": monitor_loss,
                "lr": lr_now,
            })

            improved = monitor_loss < (best_metric - min_delta)
            if improved:
                best_metric = monitor_loss
                best_state = copy.deepcopy(nets.state_dict())
                best_epoch = epoch
                bad_count = 0
            else:
                bad_count += 1

            print(
                f"[ep {epoch}] train={train_loss:.6f}  "
                f"val={monitor_loss:.6f}  lr={lr_now:.2e}  "
                f"best={best_metric:.6f} @ {best_epoch}"
            )

            if bad_count >= patience:
                print(f"Early stop at ep {epoch}, restore best epoch {best_epoch}")
                break

    nets.load_state_dict(best_state)
    return nets, history, best_epoch, best_metric
def load_countfm_checkpoint(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)

    init_cfg = ckpt["model_init"]

    if init_cfg["base_name"] != "ChunkedAdaLNTransformer":
        raise ValueError(f"Unexpected base model: {init_cfg['base_name']}")
    if init_cfg["wrapper_name"] != "ChunkedAdaLNTransformer_rate":
        raise ValueError(f"Unexpected wrapper model: {init_cfg['wrapper_name']}")

    net_base = ChunkedAdaLNTransformer(
        **init_cfg["base_kwargs"]
    ).to(device)

    nets = ChunkedAdaLNTransformer_rate(
        net_base,
        **init_cfg["wrapper_kwargs"]
    ).to(device)

    nets.load_state_dict(ckpt["model_state_dict"])
    nets.eval()

    return nets, ckpt
@torch.no_grad()
def sample_euler_batched_with_snaps(
    nets, n_step, x0_cpu, device,
    batch_eval=32, eps_t=1e-4, eps_log=1e-8, separate_heads=False,
    snap_ids=None, snap_n=0
):
    with inference_mode(nets):
        if n_step <= 0 or not 0 < eps_t < 1:
            raise ValueError("Require n_step > 0 and 0 < eps_t < 1")
        N, d = x0_cpu.shape
        snap_n = min(int(snap_n), N)
        if snap_ids is not None and any(int(s) < 0 or int(s) > n_step for s in snap_ids):
            raise ValueError("snap_ids must lie between 0 and n_step")

        if snap_ids is None or snap_n <= 0:
            snap_set = set()
            snap_ids = []
        else:
            snap_ids = sorted(set(int(s) for s in snap_ids))
            snap_set = set(snap_ids)

        snaps = {}
        if snap_set:
            for s in snap_ids:
                snaps[s] = torch.empty((snap_n, d), dtype=torch.float32, device="cpu")

        outs = []
        Delta = (1.0 - eps_t) / float(n_step)
        Delta_t = torch.tensor(Delta, device=device, dtype=torch.float32)

        for start in range(0, N, batch_eval):
            end = min(start + batch_eval, N)
            xt = x0_cpu[start:end].to(device=device, dtype=torch.float32).clone()
            B = xt.shape[0]
            t = torch.full((B, 1), 0.0, device=device, dtype=torch.float32)

            # step 0 snapshot
            if 0 in snap_set and start < snap_n:
                take = min(end, snap_n) - start
                if take > 0:
                    snaps[0][start:start+take] = xt[:take].detach().cpu()

            for k in range(1, n_step + 1):
                xt_t = torch.cat([xt, t], dim=1)

                if not separate_heads:
                    out = nets(xt_t)
                    lambda_theta, beta_theta = out[:, :d], out[:, d:]
                else:
                    lambda_theta = nets[0](xt_t)
                    beta_theta   = nets[1](xt_t)

                idx_0 = (xt <= 0).to(torch.float32)
                mu_theta = (xt * beta_theta) * (1.0 - idx_0)
                p_none, p_birth, p_death = jump_probabilities(lambda_theta, mu_theta, Delta_t)

                probs3 = torch.stack([p_none, p_birth, p_death], dim=-1)
                choice = torch.multinomial(probs3.reshape(-1, 3), 1).view(B, d)

                adj = (choice == 1).to(torch.float32) - (choice == 2).to(torch.float32)
                xt = torch.clamp(xt + adj, min=0.0)

                t = torch.minimum(t + Delta_t, torch.full_like(t, 1.0 - eps_t))
                if k in snap_set and start < snap_n:
                    take = min(end, snap_n) - start
                    if take > 0:
                        snaps[k][start:start+take] = xt[:take].detach().cpu()

            outs.append(xt.to(torch.long).cpu())

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        x1_all = torch.cat(outs, dim=0)
        return x1_all, snaps
