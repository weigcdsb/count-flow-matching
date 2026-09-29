"""Probability-path construction and plotting helpers."""

import numpy as np
import matplotlib.pyplot as plt

def build_count_dataset_np(count_array, C_max=None, margin=2):
    counts = np.asarray(count_array, dtype=np.int64)
    if C_max is None:
        C_max = int(counts.max() + margin)
    counts = np.clip(counts, 0, C_max)
    return counts, C_max


def sample_gamma_poisson_mixture(N, shapes, rates, probs=None):
    shapes = np.asarray(shapes)
    rates  = np.asarray(rates)
    K = len(shapes)
    if probs is None:
        probs = np.full(K, 1.0 / K)
    z = np.random.choice(K, size=N, p=probs)
    lam = np.random.gamma(shape=shapes[z], scale=1.0 / rates[z])
    return np.random.poisson(lam)


def sample_beta_binomial_mixture(N, n_vec, alpha_vec, beta_vec, probs=None):
    n_vec = np.asarray(n_vec, dtype=int)
    alpha_vec = np.asarray(alpha_vec, dtype=float)
    beta_vec  = np.asarray(beta_vec, dtype=float)
    K = len(n_vec)
    assert len(alpha_vec) == K and len(beta_vec) == K

    if probs is None:
        probs = np.full(K, 1.0 / K)
    probs = np.asarray(probs, dtype=float)
    
    z = np.random.choice(K, size=N, p=probs)
    
    n = n_vec[z]
    a = alpha_vec[z]
    b = beta_vec[z]
    
    p = np.random.beta(a, b)
    x = np.random.binomial(n, p)
    return x


def sample_uniform_counts(N, C_max):
    return np.random.randint(0, C_max + 1, size=N)


def binBridge_sample_xt(x0, x1, t, **kwargs):
    x0 = np.asarray(x0)
    x1 = np.asarray(x1)
    diff = x1 - x0
    sign = np.sign(diff)
    n = np.abs(diff).astype(int)
    b = np.random.binomial(n, t)
    return x0 + sign * b


def alpha_max_for_eps(K, eps = 0.01):
    return (1.0 - eps) / eps * (K - 1)


def sample_dirichlet_xt_alpha_np(y, alpha, K):
    y = np.asarray(y, dtype=int)
    B = y.shape[0]

    if np.isscalar(alpha):
        alpha_vec = np.ones((B, K), dtype=float)
        alpha_vec[np.arange(B), y] = alpha
    else:
        alpha = np.asarray(alpha)
        alpha_vec = np.ones((B, K), dtype=float)
        alpha_vec[np.arange(B), y] = alpha

    gamma = np.random.gamma(shape=alpha_vec, scale=1.0)
    probs = gamma / gamma.sum(axis=1, keepdims=True)   # [B, K]

    # sample categorical from probs, vectorized
    u = np.random.rand(B)
    cumprobs = np.cumsum(probs, axis=1)
    y_t = (u[:, None] > cumprobs).sum(axis=1)
    return y_t


def dirBridge_sample_xt(x0, x1, t, K, alpha_max, **kwargs):
    y = np.asarray(x1, dtype=int)
    scale = max(alpha_max, 1.0)
    alpha_t = 1.0 + (alpha_max - 1.0) * (t / scale)
    return sample_dirichlet_xt_alpha_np(y, alpha_t, K)


def build_time_prob_path(
        T_steps,
        M_samples,
        sample_xt,
        source_sampler,
        target_sampler,
        C_max=None,
        xt_kwargs=None,
        target_clip_margin=0,
        t0=0.0,
        t1=1.0,
        seed=None):
    if seed is not None:
        np.random.seed(seed)
    if xt_kwargs is None:
        xt_kwargs = {}

    t_grid = np.linspace(t0, t1, T_steps)
    T = T_steps

    # infer C_max from target if needed
    if C_max is None:
        tmp = target_sampler(50000)
        _, C_max = build_count_dataset_np(tmp, C_max=None, margin=target_clip_margin)

    X_t_all = np.zeros((T, M_samples), dtype=int)

    for i, t in enumerate(t_grid):
        x0 = source_sampler(M_samples, C_max)
        x1_counts = target_sampler(M_samples)
        x1, _ = build_count_dataset_np(x1_counts, C_max=C_max, margin=target_clip_margin)
        X_t_all[i] = sample_xt(x0, x1, t, **xt_kwargs)

    return X_t_all, t_grid, C_max


def entropy_from_samples(x, C_max=None):
    x = np.asarray(x, dtype=int)
    if C_max is None:
        C_max = int(x.max())
    hist = np.bincount(x, minlength=C_max + 1).astype(float)
    p = hist / hist.sum()
    p = p[p > 0]
    return -np.sum(p * np.log(p))


def invert_eta_to_t(eta_grid, t_grid, eta_rows):
    x = eta_grid[::-1]
    y = t_grid[::-1]
    eta_rows = np.asarray(eta_rows)
    eta_clipped = np.clip(eta_rows, x[0], x[-1])
    t_rows = np.interp(eta_clipped, x, y)
    return t_rows


def compute_entropy_curve(
        t_grid,
        M_entropy,
        C_max,
        sample_xt,
        source_sampler,
        target_sampler,
        xt_kwargs=None,
        target_clip_margin=0):
    if xt_kwargs is None:
        xt_kwargs = {}

    H = np.zeros_like(t_grid, dtype=float)
    for i, t in enumerate(t_grid):
        x0 = source_sampler(M_entropy, C_max)
        x1_counts = target_sampler(M_entropy)
        x1, _ = build_count_dataset_np(x1_counts, C_max=C_max, margin=target_clip_margin)
        xt = sample_xt(x0, x1, t, **xt_kwargs)
        H[i] = entropy_from_samples(xt, C_max=C_max)

    H0, H1 = H[0], H[-1]
    eta = (H - H1) / (H0 - H1 + 1e-12)
    return H, eta


def build_entropy_aligned_path(
        T_rows,
        M_samples,
        sample_xt,
        source_sampler,
        target_sampler,
        C_max,
        xt_kwargs=None,
        target_clip_margin=0,
        t0=0.0,
        t1=1.0,
        T_entropy_grid=50,
        M_entropy=None,
        seed=None):
    if seed is not None:
        np.random.seed(seed)
    if xt_kwargs is None:
        xt_kwargs = {}
    if M_entropy is None:
        M_entropy = M_samples * 20

    t_grid = np.linspace(t0, t1, T_entropy_grid)

    H, eta_grid = compute_entropy_curve(
        t_grid=t_grid,
        M_entropy=M_entropy,
        C_max=C_max,
        sample_xt=sample_xt,
        source_sampler=source_sampler,
        target_sampler=target_sampler,
        xt_kwargs=xt_kwargs,
        target_clip_margin=target_clip_margin,
    )

    eta_rows = np.linspace(0.0, 1.0, T_rows)
    t_rows = invert_eta_to_t(eta_grid, t_grid, eta_rows)
    X_t_all = np.zeros((T_rows, M_samples), dtype=int)
    for i in range(T_rows):
        t = float(t_rows[i])
        x0 = source_sampler(M_samples, C_max)
        x1_counts = target_sampler(M_samples)
        x1, _ = build_count_dataset_np(x1_counts, C_max=C_max, margin=target_clip_margin)
        X_t_all[i] = sample_xt(x0, x1, t, **xt_kwargs)

    return X_t_all, t_rows, eta_rows, H, eta_grid


def plot_count_prob_path(P, y_values=None, max_xticks=20,
                         y_label="t", invert_y=False, ax=None):
    P = np.asarray(P)
    T, N_bins = P.shape
    counts = np.arange(N_bins)

    if y_values is None:
        y_values = np.arange(T, dtype=float)
    y_values = np.asarray(y_values)

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    else:
        fig = ax.figure

    im = ax.imshow(
        P,
        origin='lower',
        aspect='auto',
        cmap='Blues',
        vmin=0.0,
        vmax=P.max() if P.max() > 0 else 1.0
    )

    # x-axis: counts, thinned if too many
    if N_bins <= max_xticks:
        xticks = np.arange(N_bins)
    else:
        step = int(np.ceil(N_bins / max_xticks))
        xticks = np.arange(0, N_bins, step)

    ax.set_xticks(xticks)
    ax.set_xticklabels(counts[xticks])
    ax.set_xlabel("Count value")

    if T <= 5:
        tick_idx = np.arange(T)
    else:
        tick_idx = np.round(np.linspace(0, T - 1, 5)).astype(int)

    ax.set_yticks(tick_idx)
    ax.set_yticklabels([f"{float(y_values[j]):.2f}" for j in tick_idx])
    ax.set_ylabel(y_label)

    if invert_y:
        ax.invert_yaxis()

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Probability")

    fig.tight_layout()
    return fig, ax


def plot_count_prob_path_from_samples(samples, t0=0.0, t1=1.0,
                                      t_grid=None, max_xticks=20, ax=None):
    samples = np.asarray(samples)
    T, N_samples = samples.shape

    max_count = int(samples.max())
    N_bins = max_count + 1

    P = np.zeros((T, N_bins), dtype=float)
    for t in range(T):
        row = samples[t].astype(int)
        hist = np.bincount(row, minlength=N_bins)
        P[t] = hist / hist.sum()

    # time values for each row
    if t_grid is not None:
        t_values = np.asarray(t_grid, dtype=float)
        if t_values.shape[0] != T:
            raise ValueError(f"t_grid length {t_values.shape[0]} != number of rows {T}")
    else:
        t_values = np.linspace(t0, t1, T)

    fig, ax = plot_count_prob_path(
        P,
        y_values=t_values,
        max_xticks=max_xticks,
        y_label="t",
        invert_y=False,
        ax=ax
    )
    return fig, ax


def plot_count_prob_path_vs_noise(samples, noise_levels, max_xticks=20,
                                  y_label="normalized entropy eta", ax=None):
    samples = np.asarray(samples)
    T, M = samples.shape

    max_count = int(samples.max())
    N_bins = max_count + 1

    P = np.zeros((T, N_bins), dtype=float)
    for t in range(T):
        row = samples[t].astype(int)
        hist = np.bincount(row, minlength=N_bins)
        P[t] = hist / hist.sum()

    fig, ax = plot_count_prob_path(
        P,
        y_values=noise_levels,
        max_xticks=max_xticks,
        y_label=y_label,
        invert_y=True,
        ax=ax
    )
    return fig, ax
