"""PCx figure checkpoint loading and original training recipe."""
import copy
import torch
from torch.utils.data import TensorDataset, DataLoader
from count_fm.models.pcx import CondCountFMNet, CondCountFMNet_MLP
from count_fm.bridge import sample_xt, sample_rt, model_loss

def extract_state_dict(blob):
    if isinstance(blob, dict):
        if "model_state_dict" in blob:
            return blob["model_state_dict"]
        if "state_dict" in blob:
            return blob["state_dict"]
    return blob


def extract_model_kwargs(blob):
    if isinstance(blob, dict) and "model_kwargs" in blob:
        return dict(blob["model_kwargs"])
    return None


def infer_countfm_architecture(state_dict, model_kwargs=None):
    if model_kwargs is not None:
        if "hidden" in model_kwargs:
            return "mlp"
        if "d_model" in model_kwargs:
            return "transformer"
    keys = list(state_dict.keys())
    if any(k.startswith("mlp.") for k in keys):
        return "mlp"
    if any(k.startswith("backbone.") for k in keys):
        return "transformer"
    raise ValueError("Cannot infer count-FM architecture from checkpoint.")


def build_countfm_model_from_blob(blob, d, cont_dim, cat_cardinalities, eps_time):
    state_dict = extract_state_dict(blob)
    model_kwargs = extract_model_kwargs(blob)
    arch = infer_countfm_architecture(state_dict, model_kwargs=model_kwargs)

    if model_kwargs is None:
        if arch == "mlp":
            model_kwargs = dict(
                d=d,
                cont_dim=cont_dim,
                cat_cardinalities=cat_cardinalities,
                emb_dim=16,
                hidden=128,
                depth=3,
                log_cap=5.0,
                eps_time=eps_time,
            )
        else:
            model_kwargs = dict(
                d=d,
                cont_dim=cont_dim,
                cat_cardinalities=cat_cardinalities,
                emb_dim=16,
                cond_dim=32,
                d_model=64,
                depth=4,
                n_heads=4,
                mlp_ratio=4.0,
                dropout=0.0,
                chunk_size=16,
                log_cap=5.0,
                eps_time=eps_time,
            )
    else:
        model_kwargs["d"] = d
        model_kwargs["cont_dim"] = cont_dim
        model_kwargs["cat_cardinalities"] = cat_cardinalities
        model_kwargs["eps_time"] = eps_time

    if arch == "mlp":
        model = CondCountFMNet_MLP(**model_kwargs)
    else:
        model = CondCountFMNet(**model_kwargs)

    model.load_state_dict(state_dict, strict=True)
    return model, arch, model_kwargs


def load_model_state(model, ckpt_path):
    blob = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(extract_state_dict(blob), strict=True)
    return model


def train_supervised(model, train_loader, val_loader, loss_fn, lr=1e-3, weight_decay=1e-4, num_epochs=40):
    device = next(model.parameters()).device
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_state = None
    best_val = float("inf")

    for epoch in range(num_epochs):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

        model.eval()
        val_sum = 0.0
        val_n = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                pred = model(xb)
                loss = loss_fn(pred, yb)
                val_sum += float(loss) * xb.shape[0]
                val_n += xb.shape[0]
        val_loss = val_sum / max(val_n, 1)

        if val_loss < best_val:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    return model


def train_countfm(
    model,
    Xtr_t, Ctr_cont_t, Ctr_cat_t,
    Xva_t, Cva_cont_t, Cva_cat_t,
    x0_poisson_rate,
    batch_size,
    lr,
    weight_decay,
    num_epochs,
    p_uncond,
    eps_t,
):
    train_ds = TensorDataset(Xtr_t, Ctr_cont_t, Ctr_cat_t)
    val_ds = TensorDataset(Xva_t, Cva_cont_t, Cva_cat_t)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)

    device = next(model.parameters()).device
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_state = None
    best_val = float("inf")

    for epoch in range(num_epochs):
        model.train()
        for x1, ccont, ccat in train_loader:
            x1 = x1.to(device)
            ccont = ccont.to(device)
            ccat = ccat.to(device)

            x0 = torch.poisson(torch.full_like(x1, float(x0_poisson_rate), dtype=torch.float32)).to(x1.dtype)
            t = torch.rand(x1.shape[0], 1, device=device) * (1.0 - eps_t)
            xt = sample_xt(x0, x1, t)
            rates_star, idx_0 = sample_rt(xt, x1, t, eps_t)

            is_uncond = torch.rand(x1.shape[0], device=device) < p_uncond
            lam, beta = model(xt, t, ccont, ccat, is_uncond_mask=is_uncond)
            mu = (xt * beta) * (1.0 - idx_0)
            rates_theta = torch.cat([lam, mu], dim=1)

            loss = model_loss("poisson", rates_theta, rates_star)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

        model.eval()
        val_sum = 0.0
        val_n = 0
        with torch.no_grad():
            for x1, ccont, ccat in val_loader:
                x1 = x1.to(device)
                ccont = ccont.to(device)
                ccat = ccat.to(device)

                x0 = torch.poisson(torch.full_like(x1, float(x0_poisson_rate), dtype=torch.float32)).to(x1.dtype)
                t = torch.rand(x1.shape[0], 1, device=device) * (1.0 - eps_t)
                xt = sample_xt(x0, x1, t)
                rates_star, idx_0 = sample_rt(xt, x1, t, eps_t)

                lam, beta = model(xt, t, ccont, ccat, is_uncond_mask=None)
                mu = (xt * beta) * (1.0 - idx_0)
                rates_theta = torch.cat([lam, mu], dim=1)

                loss = model_loss("poisson", rates_theta, rates_star)
                val_sum += float(loss) * x1.shape[0]
                val_n += x1.shape[0]

        val_loss = val_sum / max(val_n, 1)
        if val_loss < best_val:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    return model
