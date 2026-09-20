import numpy as np
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
import scipy.special as sc
from DirichletFM.utils.flow_utils import DirichletConditionalFlow, simplex_proj
import math
from models import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)

#### huristic hyperparameters
def alpha_scale_from_max(alpha_max,
                         alpha_min = 1.0,
                         tail_mass = 0.99):
    assert alpha_max > alpha_min
    assert 0.0 < tail_mass < 1.0

    t_max = alpha_max - alpha_min
    eps = 1.0 - tail_mass
    alpha_scale = t_max / (-math.log(eps))
    return alpha_scale


def suggest_alpha_hyper(C_max,
                        alpha_min=1.0,
                        mu_high=0.85,   # desired E[x_y] at largest noise
                        tail_mass=0.99  # fraction of samples below t_max
                        ):
    K = C_max + 1
    t_max = (mu_high * K - 1.0) / (1.0 - mu_high)
    if t_max <= 0:
        raise ValueError("Choose mu_high large enough so mu_high * K > 1.")

    alpha_max = alpha_min + t_max
    
    eps = 1.0 - tail_mass
    alpha_scale = t_max / (-math.log(eps))

    return alpha_min, alpha_max, alpha_scale

#### process data
def build_count_dataset(count_array, C_max=None, margin=5):
    counts = torch.as_tensor(count_array, dtype=torch.long)
    if C_max is None:
        C_max = int(counts.max().item() + margin)
    counts = counts.clamp_min(0).clamp_max(C_max)
    return counts, C_max

#### classifier
class DirichletFMClassifier(nn.Module):
    def __init__(self, num_dims, K, mlp_width=64, time_varying=True):
        super().__init__()
        self.D = num_dims
        self.K = K
        self.mlp = MLP(dim=num_dims * K,
                       out_dim=num_dims * K,
                       w=mlp_width,
                       time_varying=time_varying)

    def forward(self, x_t, alpha):
        B, D, K = x_t.shape
        assert D == self.D and K == self.K

        x_flat = x_t.reshape(B, D * K)
        if not torch.is_tensor(alpha):
            alpha = torch.full((B, 1), float(alpha), device=x_t.device)
        else:
            alpha = alpha.to(device=x_t.device)
            if alpha.ndim == 0:
                alpha = alpha.view(1, 1).expand(B, 1)
            elif alpha.ndim == 1:
                alpha = alpha.view(B, 1)

        if getattr(self.mlp, "time_varying", False):
            x_in = torch.cat([x_flat, alpha], dim=1)
        else:
            x_in = x_flat

        logits_flat = self.mlp(x_in)
        return logits_flat.view(B, D, K)


#### Dirichlet FM
def sample_dirichlet_xt_alpha(y, alpha, K):
    device = y.device
    B, D = y.shape
    if not torch.is_tensor(alpha):
        alpha = torch.full((B,), float(alpha), device=device)
    alpha = alpha.view(B, 1, 1)

    y_onehot = F.one_hot(y, num_classes=K).float()
    alphas_full = torch.ones(B, D, K, device=device) + y_onehot * (alpha - 1.0)
    dist = torch.distributions.Dirichlet(alphas_full)
    x_t = dist.rsample()
    return x_t


def dirichlet_fm_loss_alpha(classifier, counts_batch, K,
                            alpha_scale=1.0,
                            alpha_min=1.0,
                            alpha_max=100.0,
                            schedule = 'exp'):
    device = counts_batch.device
    B, D = counts_batch.shape

    if schedule == 'exp':
        t_raw = torch.distributions.Exponential(1.0).sample((B,)).to(device)
        t = t_raw * alpha_scale
        alpha = 1.0 + t
    else:
        alpha = torch.rand(B, device=device) * (alpha_max - alpha_min) + alpha_min

    # keep within table used by condflow
    alpha = alpha.clamp(max=alpha_max)

    x_t = sample_dirichlet_xt_alpha(counts_batch, alpha, K)
    logits = classifier(x_t, alpha)

    loss = F.cross_entropy(
        logits.view(B * D, K),
        counts_batch.view(B * D),
        reduction="mean",
    )
    return loss


@torch.no_grad()
def dirichlet_vector_field_alpha(x, alpha, classifier, condflow):
    device = x.device
    B, D, K = x.shape
    
    alpha_batch = torch.full((B,), float(alpha), device=device)
    logits = classifier(x, alpha_batch)
    p_hat = F.softmax(logits, dim=-1)
    
    x_cpu = x.detach().cpu().numpy()
    x_cpu = np.clip(x_cpu, 1e-9, 1.0 - 1e-9)
    C_np = condflow.c_factor(x_cpu, float(alpha))
    C = torch.from_numpy(C_np).to(device=x.device, dtype=x.dtype)
    
    eye = torch.eye(K, device=device)
    cond_flows = (eye - x.unsqueeze(-1)) * C.unsqueeze(-2)
    v = (p_hat.unsqueeze(-2) * cond_flows).sum(dim=-1)
    return v



def DFM_train(train_counts, C_max, batch_size, num_epochs,
              classifier, alpha_scale, alpha_min, alpha_max, t_schedule,
              opt, device):

    N, D = train_counts.shape
    steps_per_epoch = (N + batch_size - 1) // batch_size
    K = C_max + 1
    
    for epoch in tqdm(range(num_epochs)):
        for step in range(steps_per_epoch):
    
            # 1. sample data batch (analog of x1 in FM)
            idx = torch.randint(0, N, (batch_size,), device=device)
            batch = train_counts[idx]   # [B, D]
    
            # 2–4. DFM loss (Dirichlet noise + classifier)
            loss = dirichlet_fm_loss_alpha(classifier, batch, K,
                                           alpha_scale,
                                           alpha_min,
                                           alpha_max,
                                           t_schedule)
    
            # 5–7. backprop (same optimizer, lr, etc. as FM)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    
        # 8. logging, matched to FM style
        if epoch % max(1, (num_epochs // 5)) == 0:
            print(f"[DFM][epoch {epoch}] loss={float(loss):.6f}")

    return classifier, loss


#### generation
@torch.no_grad()
def sample_dfm_counts(classifier,
                      condflow,
                      num_samples,
                      D,
                      C_max,
                      num_steps=100,
                      alpha_min=1.0,
                      alpha_max=100.0,
                      device="cuda",
                      return_traj=False,
                      x0=None):
    device = torch.device(device)
    K = C_max + 1

    if x0 is None:
        B = num_samples
        dist0 = torch.distributions.Dirichlet(torch.ones(K, device=device))
        x = dist0.rsample((B, D))     # [B,D,K]
        x = x / x.sum(-1, keepdim=True)
    else:
        x = x0.to(device)
        B, D_x, K_x = x.shape

    traj = []
    if return_traj:
        traj.append(x.detach().cpu())   # [B,D,K]

    # integration grid in α
    alphas = torch.linspace(alpha_min, alpha_max, num_steps + 1, device=device)

    for i in range(num_steps):
        s = alphas[i].item()
        t = alphas[i + 1].item()
        dt = t - s

        v = dirichlet_vector_field_alpha(x, s, classifier, condflow)  # [B,D,K]
        x = x + dt * v

        # keep on simplex
        x = x.clamp_min(1e-9)
        x = x / x.sum(-1, keepdim=True)

        if return_traj:
            traj.append(x.detach().cpu())  # [B,D,K]

    counts_idx = x.argmax(-1)   # [B,D]
    counts = counts_idx.clamp_max(C_max)

    if return_traj:
        traj_probs = torch.stack(traj, dim=0)
        return counts.cpu(), traj_probs
    else:
        return counts.cpu()







