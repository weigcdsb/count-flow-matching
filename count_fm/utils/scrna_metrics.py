"""Training-fitted scRNA features, Gaussian W2 and RBF MMD."""
import numpy as np
import torch
from sklearn.decomposition import PCA

def log1p_libnorm_np(X_counts, target_sum=1e4):
    X_counts = X_counts.astype(np.float32, copy=False)
    lib = X_counts.sum(axis=1, keepdims=True)
    lib = np.maximum(lib, 1.0).astype(np.float32)
    Xn = X_counts * (float(target_sum) / lib)
    return np.log1p(Xn)


class Featurizer:
    def __init__(self, target_sum=1e4, g_keep=256, pca_dim=50, seed=0):
        self.target_sum = target_sum
        self.g_keep = g_keep
        self.pca_dim = pca_dim
        self.seed = seed
        self.top_idx = None
        self.pca = None

    def fit(self, X_real_counts):
        X_log = log1p_libnorm_np(X_real_counts, self.target_sum)
        gene_var = X_log.var(axis=0)
        g_keep = min(self.g_keep, X_log.shape[1])
        self.top_idx = np.argsort(gene_var)[-g_keep:]

        Z = X_log[:, self.top_idx]
        if self.pca_dim is None:
            self.pca = None
            return self

        p = min(self.pca_dim, Z.shape[1], Z.shape[0] - 1)
        self.pca = PCA(n_components=p, svd_solver="randomized", random_state=self.seed)
        self.pca.fit(Z)
        return self

    def transform(self, X_counts):
        X_log = log1p_libnorm_np(X_counts, self.target_sum)
        Z = X_log[:, self.top_idx]
        if self.pca is None:
            return Z
        return self.pca.transform(Z)
def _cov_np(Z):
    Z = np.asarray(Z, dtype=np.float64)
    mu = Z.mean(axis=0)
    Xc = Z - mu[None, :]
    cov = (Xc.T @ Xc) / max(Z.shape[0] - 1, 1)
    return mu, cov


def _sqrtm_psd(A, eps=1e-12):
    w, V = np.linalg.eigh(A)
    w = np.clip(w, 0.0, None)
    return (V * np.sqrt(w + eps)) @ V.T


def w2_gaussian(Z_real, Z_fake):
    mu1, C1 = _cov_np(Z_real)
    mu2, C2 = _cov_np(Z_fake)

    diff = mu1 - mu2
    diff2 = float(diff @ diff)

    sqrtC1 = _sqrtm_psd(C1)
    A = sqrtC1 @ C2 @ sqrtC1
    sqrtA = _sqrtm_psd(A)

    tr = np.trace(C1 + C2 - 2.0 * sqrtA)
    w2_sq = diff2 + float(tr)
    w2_sq = max(w2_sq, 0.0)
    return float(np.sqrt(w2_sq))


def _median_heuristic_sigma2(Z, max_points=1000, seed=0):
    rng = np.random.default_rng(seed)
    n = Z.shape[0]
    m = min(n, max_points)
    idx = rng.choice(n, size=m, replace=False)
    X = torch.tensor(Z[idx], dtype=torch.float32)
    x2 = (X**2).sum(dim=1, keepdim=True)
    dist2 = x2 + x2.T - 2 * (X @ X.T)
    dist2 = dist2.flatten()
    dist2 = dist2[dist2 > 0]
    return float(torch.median(dist2).item())


def mmd2_rbf_unbiased(Z_real, Z_fake, sigma2=None, device=None, seed=0):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    X = torch.tensor(Z_real, dtype=torch.float32, device=device)
    Y = torch.tensor(Z_fake, dtype=torch.float32, device=device)

    m = X.shape[0]
    n = Y.shape[0]

    if sigma2 is None:
        sigma2 = _median_heuristic_sigma2(np.vstack([Z_real, Z_fake]), seed=seed)
        sigma2 = max(sigma2, 1e-6)

    def k_rbf(A, B):
        a2 = (A**2).sum(dim=1, keepdim=True)
        b2 = (B**2).sum(dim=1, keepdim=True).T
        dist2 = a2 + b2 - 2.0 * (A @ B.T)
        return torch.exp(-dist2 / (2.0 * sigma2))

    Kxx = k_rbf(X, X)
    Kyy = k_rbf(Y, Y)
    Kxy = k_rbf(X, Y)

    Kxx.fill_diagonal_(0.0)
    Kyy.fill_diagonal_(0.0)

    term_x = Kxx.sum() / (m * (m - 1))
    term_y = Kyy.sum() / (n * (n - 1))
    term_xy = Kxy.mean()

    mmd2 = term_x + term_y - 2.0 * term_xy
    return float(mmd2.detach().cpu().item()), float(sigma2)
