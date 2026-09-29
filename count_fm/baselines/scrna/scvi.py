"""Original SCVIModel adapter; requires the upstream baseline environment."""
import torch
import anndata as ad
import scvi
import math
import numpy as np
import os
from count_fm.experiments.base import UncondModel

class SCVIModel(UncondModel):
    name = "scVI"
    def __init__(self, device="cuda", n_latent=10, n_layers=2, n_hidden=128,
                 gene_likelihood="zinb", max_epochs=400):
        self.device = device
        self.n_latent = n_latent
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.gene_likelihood = gene_likelihood
        self.max_epochs = max_epochs
        self.model = None

        self.lr = 1e-3
        self.weight_decay = 1e-6

    def fit(self, X_train_counts, lr=None, weight_decay=None, max_epochs=None, resume=True):
        lr = float(self.lr if lr is None else lr)
        weight_decay = float(self.weight_decay if weight_decay is None else weight_decay)
        max_epochs = int(self.max_epochs if max_epochs is None else max_epochs)

        if (not resume) or (self.model is None):
            adata_train = ad.AnnData(X_train_counts.astype(np.int64))
            scvi.model.SCVI.setup_anndata(adata_train)
            self.model = scvi.model.SCVI(
                adata_train,
                n_latent=self.n_latent,
                n_layers=self.n_layers,
                n_hidden=self.n_hidden,
                gene_likelihood=self.gene_likelihood,
                use_observed_lib_size=False,
            )

        try:
            self.model.train(
                max_epochs=max_epochs,
                accelerator="gpu" if (self.device == "cuda" and torch.cuda.is_available()) else "cpu",
                devices=1,
                plan_kwargs={"lr": lr, "weight_decay": weight_decay},
            )
        except TypeError:
            self.model.train(
                max_epochs=max_epochs,
                accelerator="gpu" if (self.device == "cuda" and torch.cuda.is_available()) else "cpu",
                devices=1,
                lr=lr,
                weight_decay=weight_decay,
            )

        print(f"scVI: resume={resume} lr={lr} wd={weight_decay} epochs={max_epochs}")
        return self

    @torch.no_grad()
    def sample(self, n_samples):
        assert self.model is not None
        m = self.model
        module = m.module
        dev = m.device
    
        # latent prior
        n_latent = getattr(module, "n_latent", None)
        if n_latent is None:
            n_latent = self.n_latent
        z = torch.randn((n_samples, int(n_latent)), device=dev)
    
        # batch index (Dentate Gyrus is single batch in your setup)
        batch_index = torch.zeros((n_samples, 1), dtype=torch.long, device=dev)
    
        # library prior (works when use_observed_lib_size=False)
        lib_means = getattr(module, "library_log_means", None)
        lib_vars  = getattr(module, "library_log_vars", None)
    
        if lib_means is not None and lib_vars is not None:
            lib_means = torch.as_tensor(lib_means, dtype=torch.float32, device=dev).view(-1)
            lib_vars  = torch.as_tensor(lib_vars,  dtype=torch.float32, device=dev).view(-1)
    
            # pick batch-specific params (here always 0)
            b = batch_index.view(-1)
            mu  = lib_means[b].view(-1, 1)
            var = lib_vars[b].view(-1, 1)
            library = torch.randn((n_samples, 1), device=dev) * torch.sqrt(var + 1e-8) + mu
        else:
            # fallback: empirical log library from training data
            Xtr = m.adata.X
            if hasattr(Xtr, "toarray"):
                Xtr = Xtr.toarray()
            log_lib = np.log(np.maximum(np.asarray(Xtr).sum(axis=1), 1.0))
            mu = float(np.mean(log_lib))
            var = float(np.var(log_lib))
            library = torch.randn((n_samples, 1), device=dev) * math.sqrt(var + 1e-8) + mu
    
        # decode and sample counts
        outs = module.generative(z=z, library=library, batch_index=batch_index)
        px = outs["px"]
        x = px.sample()
    
        x = torch.clamp(x, min=0.0)
        x = x.detach().cpu().numpy()
        return np.rint(x).astype(np.int64)
def save_scvi(scvi_wrapper, path_dir):
    os.makedirs(path_dir, exist_ok=True)
    # critical: save AnnData so SCVI.load(path_dir) works
    scvi_wrapper.model.save(path_dir, overwrite=True, save_anndata=True)
    with open(os.path.join(path_dir, "scvi_tools_version.txt"), "w") as f:
        f.write(scvi.__version__)
    print("Saved scVI to", path_dir, "(save_anndata=True)")


def load_scvi(path_dir, device="cuda", X_ref_counts=None):
    """
    PyTorch 2.6: torch.load defaults weights_only=True, which breaks scvi-tools loading.
    Since this is YOUR checkpoint, we force weights_only=False when needed.
    """
    def _do_load():
        try:
            return scvi.model.SCVI.load(path_dir)
        except Exception as e:
            msg = str(e).lower()
            if ("anndata" in msg) or ("adata" in msg):
                if X_ref_counts is None:
                    raise RuntimeError(
                        "SCVI.load failed because no AnnData was saved. "
                        "Pass X_ref_counts once, then re-save with save_anndata=True."
                    ) from e
                adata_tmp = ad.AnnData(np.asarray(X_ref_counts).astype(np.int64))
                return scvi.model.SCVI.load(path_dir, adata=adata_tmp)
            raise

    try:
        model = _do_load()
    except Exception as e:
        # PyTorch 2.6 weights_only failure: retry with weights_only=False (trusted checkpoint)
        if ("weights only load failed" in str(e).lower()) or ("weights_only" in str(e).lower()):
            _torch_load = torch.load
            def _torch_load_force(*args, **kwargs):
                kwargs.setdefault("weights_only", False)
                return _torch_load(*args, **kwargs)
            torch.load = _torch_load_force
            try:
                model = _do_load()
            finally:
                torch.load = _torch_load
        else:
            raise

    # move to device
    if device == "cuda" and torch.cuda.is_available():
        model.to_device("cuda:0")
    else:
        model.to_device("cpu")

    wrapper = SCVIModel(device=device)
    wrapper.model = model
    print("Loaded scVI from", path_dir)
    return wrapper
