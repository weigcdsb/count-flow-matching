"""hc-3 training, CFG sampling and checkpoint helpers."""
import torch
from torch.nn import functional as F
from tqdm.auto import tqdm
from count_fm.bridge import sample_xt, sample_rt
from count_fm.utils.inference import inference_mode
from count_fm.sampling import jump_probabilities

def countfm_loss(rates_theta, rates_star, eps=1e-8):
    u = rates_star
    v = rates_theta
    return (v - u * torch.log(v + eps)).sum(dim=1).mean()


def poisson_loss(rate, x, eps=1e-8):
    return (rate - x * torch.log(rate + eps)).sum(dim=1).mean()
def train_mean_mlp(net, X, y, epochs=200, batch_size=512, lr=1e-3):
    net.train()
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    n = X.shape[0]
    for ep in tqdm(range(epochs), desc="MLP mean"):
        perm = torch.randperm(n, device=X.device)
        for s in range(0, n, batch_size):
            idx = perm[s:s+batch_size]
            pred = F.softplus(net(y[idx]))
            loss = F.mse_loss(pred, X[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    return net


def train_poisson_mlp(net, X, y, epochs=200, batch_size=512, lr=1e-3):
    net.train()
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    n = X.shape[0]
    for ep in tqdm(range(epochs), desc="Poisson MLP"):
        perm = torch.randperm(n, device=X.device)
        for s in range(0, n, batch_size):
            idx = perm[s:s+batch_size]
            rate = net(y[idx])
            loss = poisson_loss(rate, X[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    return net


def train_countfm_cond(net, X, y, epochs=200, batch_size=512, lr=1e-3,
                       x0_rate=None, eps_t=1e-4, p_uncond=0.1, lambda_mean=0.1):
    net.train()
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    n, d = X.shape

    if x0_rate is None:
        x0_rate = float(X.mean().item())

    for ep in tqdm(range(epochs), desc="count-FM"):
        perm = torch.randperm(n, device=X.device)
        for s in range(0, n, batch_size):
            idx = perm[s:s+batch_size]
            x1 = X[idx]
            yb = y[idx]
            bsz = x1.shape[0]

            x0 = torch.poisson(torch.full_like(x1, x0_rate))

            keep_mask = (torch.rand(bsz, 1, device=X.device) > p_uncond).float()
            y_drop = yb * keep_mask
            y_cfg = torch.cat([y_drop, keep_mask], dim=1)

            t = torch.rand(bsz, 1, device=X.device) * (1.0 - eps_t)
            xt = sample_xt(x0, x1, t)
            rates_star, idx_0 = sample_rt(xt, x1, t, eps_t)

            inp = torch.cat([xt, t, y_cfg], dim=1)
            out = net(inp)

            lam = out[:, :d]
            beta = out[:, d:2*d]
            mu = (xt * beta) * (1.0 - idx_0)
            rates_theta = torch.cat([lam, mu], dim=1)

            loss_rate = countfm_loss(rates_theta, rates_star)

            # auxiliary mean loss, always on the fully conditioned branch
            y_full = torch.cat([yb, torch.ones_like(keep_mask)], dim=1)
            inp_full = torch.cat([xt, t, y_full], dim=1)
            out_full = net(inp_full)
            mu_hat = out_full[:, 2*d:3*d]
            loss_mean = poisson_loss(mu_hat, x1)

            loss = loss_rate + lambda_mean * loss_mean

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    return net
def save_hc3_checkpoint(path, mlp_mean, pois_mlp, fm_net, x0_rate, meta=None):
    payload = {
        "mlp_mean_state": mlp_mean.state_dict(),
        "pois_mlp_state": pois_mlp.state_dict(),
        "fm_net_state": fm_net.state_dict(),
        "x0_rate": float(x0_rate),
        "meta": {} if meta is None else meta,
    }
    torch.save(payload, path)
    print(f"saved checkpoint to {path}")


def load_hc3_checkpoint(path, mlp_mean, pois_mlp, fm_net, map_location=None):
    payload = torch.load(path, map_location=map_location)
    mlp_mean.load_state_dict(payload["mlp_mean_state"])
    pois_mlp.load_state_dict(payload["pois_mlp_state"])
    fm_net.load_state_dict(payload["fm_net_state"])
    x0_rate = float(payload["x0_rate"])
    meta = payload.get("meta", {})
    print(f"loaded checkpoint from {path}")
    return mlp_mean, pois_mlp, fm_net, x0_rate, meta
@torch.no_grad()
def sample_countfm_cfg_batch(net, y, d, x0_rate, n_rep=16, n_step=1000, w=1.0, eps_t=1e-4):
    with inference_mode(net):
        if n_step <= 0 or not 0 < eps_t < 1:
            raise ValueError("Require n_step > 0 and 0 < eps_t < 1")
        n = y.shape[0]
        c_dim = y.shape[1]

        y_rep = y.unsqueeze(0).expand(n_rep, n, c_dim).reshape(n_rep * n, c_dim)

        keep_c = torch.ones((n_rep * n, 1), device=y.device)
        keep_u = torch.zeros((n_rep * n, 1), device=y.device)

        y_c = torch.cat([y_rep, keep_c], dim=1)
        y_u = torch.cat([torch.zeros_like(y_rep), keep_u], dim=1)

        xt = torch.poisson(torch.full((n_rep * n, d), x0_rate, device=y.device)).float()
        t = torch.full((n_rep * n, 1), 0.0, device=y.device)
        delta = (1.0 - eps_t) / n_step

        for _ in range(n_step):
            inp_c = torch.cat([xt, t, y_c], dim=1)
            inp_u = torch.cat([xt, t, y_u], dim=1)

            out_c = net(inp_c)
            out_u = net(inp_u)

            lam_c, beta_c = out_c[:, :d], out_c[:, d:2*d]
            lam_u, beta_u = out_u[:, :d], out_u[:, d:2*d]

            lam = torch.clamp(lam_u + w * (lam_c - lam_u), min=1e-8)
            beta = torch.clamp(beta_u + w * (beta_c - beta_u), min=1e-8)

            idx0 = (xt <= 0).float()
            mu = (xt * beta) * (1.0 - idx0)

            r = lam + mu
            p_none, p_birth, p_death = jump_probabilities(lam, mu, delta)

            u = torch.rand_like(r)
            birth_mask = (u >= p_none) & (u < p_none + p_birth)
            death_mask = (u >= p_none + p_birth)

            xt = torch.clamp(xt + birth_mask.float() - death_mask.float(), min=0.0)
            t = torch.clamp(t + delta, max=1.0 - eps_t)

        return xt.view(n_rep, n, d).long()
