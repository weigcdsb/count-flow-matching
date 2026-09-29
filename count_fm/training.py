"""Count-FM training for independent or OT-coupled endpoints."""

import torch
from tqdm.auto import tqdm
from .bridge import sample_xt, sample_rt, model_forward, model_loss
from .coupling import _soft_ot_pair_batch

def CountFM_train(
    X1_torch,
    nets,
    optimizer,
    num_epochs,
    batch_size,
    device,
    separate_heads=False,
    X0_torch=None,
    x0_mode="uniform",  # "uniform", "poisson", or "dataset"
    ot_cost="none",     # "none", "L1", "L2", "sym_poisson"
    ot_pair_batch_size=None,   # NEW: candidate batch size for OT
    C_max=None,
    margin=2,
    eps_t=1e-4,
    eps_log=1e-8,
    loss_mode="poisson",
    lam0_train=1.0,
):
    if num_epochs <= 0 or batch_size <= 0 or not 0 < eps_t < 1:
        raise ValueError("Require positive num_epochs/batch_size and 0 < eps_t < 1")
    for net in (nets if separate_heads else [nets]):
        net.train()
    X1_torch = X1_torch.to(device=device, dtype=torch.float32)
    N1, d = X1_torch.shape

    if X0_torch is not None:
        X0_torch = X0_torch.to(device=device, dtype=torch.float32)
        N0 = X0_torch.shape[0]
    else:
        N0 = N1

    if C_max is None:
        C_max = int(X1_torch.max().item() + margin)

    if ot_pair_batch_size is None:
        ot_pair_batch_size = batch_size

    steps_per_epoch = (max(N0, N1) + batch_size - 1) // batch_size

    for epoch in tqdm(range(num_epochs)):
        for step in range(steps_per_epoch):
            B_pair = int(ot_pair_batch_size)

            idx1 = torch.randint(0, N1, (B_pair,), device=device)
            x1_big = X1_torch[idx1]

            if X0_torch is not None and x0_mode == "dataset":
                idx0 = torch.randint(0, N0, (B_pair,), device=device)
                x0_big = X0_torch[idx0]
            elif x0_mode == "uniform":
                x0_big = torch.randint(
                    low=0,
                    high=C_max + 1,
                    size=x1_big.shape,
                    device=device,
                    dtype=x1_big.dtype,
                )
            elif x0_mode == "poisson":
                x0_big = torch.poisson(
                    torch.full_like(x1_big, float(lam0_train), dtype=torch.float32)
                ).to(x1_big.dtype)
            else:
                raise ValueError(f"Unknown x0_mode: {x0_mode}")

            # OT is solved on the larger candidate batch, then sampled back to optimizer batch_size
            x0, x1 = _soft_ot_pair_batch(
                x0_big,
                x1_big,
                ot_cost=ot_cost,
                ot_eps=eps_log,
                out_size=batch_size,
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
            optimizer.step()

        if epoch % max(1, (num_epochs // 5)) == 0:
            print(f"[CountFM][epoch {epoch}] loss={float(loss):.6f}")

    return nets, loss
