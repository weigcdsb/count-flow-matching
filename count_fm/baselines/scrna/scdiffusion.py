"""Original ScDiffusionModel adapter; requires the upstream baseline environment."""
import os, sys, glob, re, subprocess, time
import numpy as np
import scipy.sparse as sp
import anndata as ad
import torch
import json
from count_fm.experiments.base import UncondModel

class ScDiffusionModel(UncondModel):
    name = "scDiffusion"

    def __init__(
        self,
        scd_root="scDiffusion",
        workdir="saved_models/scdiffusion_dg",
        DG_H5AD=None,                  # optional reference h5ad for var_names
        scimilarity_dir=None,
        device="cuda",
        seed=0,
        ae_steps=150000,
        diff_steps=600000,
        batch_size=128,
        ae_ckpt_freq=50000,
        diff_save_interval=200000,
        lr=1e-4,
        weight_decay=1e-4,
        latent_dim=128,
        hidden_dim=(512, 512, 256, 128),
        noise_schedule="linear",
        diffusion_steps=1000,
    ):
        self.scd_root = scd_root
        self.workdir = workdir
        self.DG_H5AD = DG_H5AD
        self.scimilarity_dir = scimilarity_dir
        self.device = device
        self.seed = int(seed)

        self.ae_steps = int(ae_steps)
        self.diff_steps = int(diff_steps)
        self.batch_size = int(batch_size)
        self.ae_ckpt_freq = int(ae_ckpt_freq)
        self.diff_save_interval = int(diff_save_interval)

        self.lr = float(lr)
        self.weight_decay = float(weight_decay)

        self.latent_dim = int(latent_dim)
        self.hidden_dim = list(hidden_dim)
        self.noise_schedule = str(noise_schedule)
        self.diffusion_steps = int(diffusion_steps)

        # full gene dim (your notebook X_train dim)
        self.d_full = None
        # filtered gene dim after scDiffusion filter_genes/filter_cells
        self.d = None

        self.keep_gene_idx = None
        self.keep_cell_idx = None

        self.data_h5ad = None
        self.ae_ckpt = None
        self.backbone_ckpt = None
        self._train_lib_sizes = None

    def _ensure_repo_on_path(self):
        root = os.path.abspath(self.scd_root)
        if root not in sys.path:
            sys.path.insert(0, root)

    @staticmethod
    def _latest_step(paths, pat=r"step=(\d+)\.pt"):
        best = (-1, None)
        for p in paths:
            m = re.search(pat, os.path.basename(p))
            if m:
                s = int(m.group(1))
                if s > best[0]:
                    best = (s, p)
        return best  # (step, path)

    @staticmethod
    def _latest_model_step(paths, pat=r"model(\d+)\.pt"):
        best = (-1, None)
        for p in paths:
            m = re.search(pat, os.path.basename(p))
            if m:
                s = int(m.group(1))
                if s > best[0]:
                    best = (s, p)
        return best  # (step, path)

    def _write_train_h5ad(self, X_train_counts, h5ad_path):
        os.makedirs(os.path.dirname(h5ad_path), exist_ok=True)

        Xc = np.asarray(X_train_counts)
        Xc = np.rint(Xc).astype(np.int64, copy=False)
        Xsp = sp.csr_matrix(Xc)
        adata_local = ad.AnnData(Xsp)

        # optional: preserve gene names (helps reproducibility)
        if self.DG_H5AD is not None:
            import scanpy as sc
            ad0 = sc.read_h5ad(self.DG_H5AD)
            ad0.var_names_make_unique()
            if ad0.shape[1] == adata_local.shape[1]:
                adata_local.var_names = ad0.var_names

        adata_local.obs["celltype"] = "all"
        adata_local.write_h5ad(h5ad_path)
        return h5ad_path

    def _infer_scdiff_filter_indices(self):
        import scanpy as sc

        ad0 = sc.read_h5ad(self.data_h5ad)
        ad0.var_names_make_unique()

        orig_var = ad0.var_names.copy()
        orig_obs = ad0.obs_names.copy()

        # match scDiffusion loader behavior
        sc.pp.filter_genes(ad0, min_cells=3)
        sc.pp.filter_cells(ad0, min_genes=10)
        ad0.var_names_make_unique()

        keep_var = ad0.var_names
        keep_obs = ad0.obs_names

        keep_gene_idx = orig_var.get_indexer(keep_var)
        keep_cell_idx = orig_obs.get_indexer(keep_obs)

        if np.any(keep_gene_idx < 0) or np.any(keep_cell_idx < 0):
            raise RuntimeError("Failed to map filtered genes/cells back to original indices.")

        self.keep_gene_idx = keep_gene_idx.astype(int)
        self.keep_cell_idx = keep_cell_idx.astype(int)
        self.d = int(len(self.keep_gene_idx))

        print("[scDiff] d_full =", self.d_full,
              "d_after_filter =", self.d,
              "n_train_full =", len(orig_obs),
              "n_after_filter =", len(self.keep_cell_idx))

    def _train_or_resume_ae(self, ae_dir, resume=True):
        self._ensure_repo_on_path()
        from guided_diffusion.cell_datasets_loader import load_data
        from VAE.VAE_model import VAE

        dev = torch.device("cuda" if (self.device == "cuda" and torch.cuda.is_available()) else "cpu")
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        os.makedirs(ae_dir, exist_ok=True)

        ae_pat = os.path.join(ae_dir, f"model_seed={self.seed}_step=*.pt")
        ae_cands = glob.glob(ae_pat)
        last_step, last_path = self._latest_step(ae_cands, pat=r"step=(\d+)\.pt")

        target_step = self.ae_steps - 1
        if resume and last_step >= target_step:
            self.ae_ckpt = last_path
            return

        # IMPORTANT: num_genes must match filtered gene count (self.d)
        vae = VAE(
            num_genes=int(self.d),
            device=str(dev),
            seed=self.seed,
            loss_ae="mse",
            hidden_dim=self.latent_dim,
            decoder_activation="ReLU",
        )

        # force fp32 batches (loader can yield float64)
        _orig_train_step = vae.train_step
        def _train_step_fp32(genes):
            if isinstance(genes, torch.Tensor) and genes.dtype != torch.float32:
                genes = genes.to(dtype=torch.float32)
            return _orig_train_step(genes)
        vae.train_step = _train_step_fp32

        if resume and last_step >= 0:
            vae.load_state_dict(torch.load(last_path, map_location="cpu"))
            start = last_step + 1
        else:
            if self.scimilarity_dir is not None:
                use_gpu = (dev.type == "cuda")
                vae.encoder.load_state(os.path.join(self.scimilarity_dir, "encoder.ckpt"), use_gpu=use_gpu)
                vae.decoder.load_state(os.path.join(self.scimilarity_dir, "decoder.ckpt"), use_gpu=use_gpu)
            else:
                print("[AE] training from scratch (no scimilarity init)")
            start = 0

        data = load_data(data_dir=self.data_h5ad, batch_size=self.batch_size, train_vae=True)
        freq = max(1, min(self.ae_ckpt_freq, target_step))

        t0 = time.time()
        for step in range(start, target_step + 1):
            genes, _ = next(data)
            stats = vae.train_step(genes)

            if step == start:
                try:
                    print("[AE] first batch shape:", tuple(genes.shape), "dtype:", genes.dtype)
                except Exception:
                    pass

            if step % 1000 == 0:
                print("[AE] step", step, "loss", stats["loss_reconstruction"], "elapsed", round(time.time()-t0, 1), "sec")

            if (step % freq == 0) or (step == target_step):
                torch.save(vae.state_dict(), os.path.join(ae_dir, f"model_seed={self.seed}_step={step}.pt"))

        self.ae_ckpt = os.path.join(ae_dir, f"model_seed={self.seed}_step={target_step}.pt")

    def fit(self, X_train_counts, lr=None, weight_decay=None, resume=True):
        self._ensure_repo_on_path()
        lr = float(self.lr if lr is None else lr)
        weight_decay = float(self.weight_decay if weight_decay is None else weight_decay)

        Xc = np.asarray(X_train_counts)
        self.d_full = int(Xc.shape[1])

        os.makedirs(self.workdir, exist_ok=True)
        self.data_h5ad = self._write_train_h5ad(
            X_train_counts,
            os.path.join(self.workdir, "train_for_scdiffusion.h5ad"),
        )

        self._infer_scdiff_filter_indices()

        # library sizes for sampling computed on kept cells + kept genes
        X_kept = Xc[self.keep_cell_idx][:, self.keep_gene_idx]
        self._train_lib_sizes = np.rint(X_kept.sum(axis=1)).astype(np.int64)

        ae_dir = os.path.join(self.workdir, "AE")
        backbone_dir = os.path.join(self.workdir, "backbone")
        model_name = "dg_scdiffusion"
        out_dir = os.path.join(backbone_dir, model_name)
        os.makedirs(out_dir, exist_ok=True)

        self._train_or_resume_ae(ae_dir, resume=resume)

        target_model = os.path.join(out_dir, f"model{self.diff_steps:06d}.pt")
        if resume and os.path.exists(target_model):
            self.backbone_ckpt = target_model
            return self

        resume_ckpt = ""
        if resume:
            cand = glob.glob(os.path.join(out_dir, "model*.pt"))
            last_step, last_path = self._latest_model_step(cand, pat=r"model(\d+)\.pt")
            if last_step >= 0 and last_step < self.diff_steps:
                resume_ckpt = last_path

        cmd = [
            sys.executable,
            os.path.join(self.scd_root, "cell_train.py"),
            "--data_dir", self.data_h5ad,
            "--vae_path", self.ae_ckpt,
            "--model_name", model_name,
            "--save_dir", backbone_dir,
            "--batch_size", str(self.batch_size),
            "--lr", str(lr),
            "--weight_decay", str(weight_decay),
            "--lr_anneal_steps", str(self.diff_steps),
            "--save_interval", str(self.diff_save_interval),
            "--input_dim", str(self.latent_dim),
            "--noise_schedule", str(self.noise_schedule),
            "--diffusion_steps", str(self.diffusion_steps),
        ]
        if resume_ckpt:
            cmd += ["--resume_checkpoint", resume_ckpt]

        print("[Diffusion] running:")
        print(" ".join(cmd))
        
        # subprocess.run(cmd, check=True)
        log_path = os.path.join(self.workdir, "diffusion_train.log")
        with open(log_path, "w") as f:
            p = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, text=True)
            print("Diffusion running, PID =", p.pid, "log =", log_path)
            ret = p.wait()
            if ret != 0:
                raise RuntimeError(f"scDiffusion diffusion stage failed, see {log_path}")
        

        if os.path.exists(target_model):
            self.backbone_ckpt = target_model
        else:
            cand = glob.glob(os.path.join(out_dir, "model*.pt"))
            _, last_path = self._latest_model_step(cand, pat=r"model(\d+)\.pt")
            if last_path is None:
                raise FileNotFoundError("No diffusion checkpoint found in " + out_dir)
            self.backbone_ckpt = last_path

        return self

    @torch.no_grad()
    def sample(self, n_samples, batch_size=1000):
        self._ensure_repo_on_path()
        if self.ae_ckpt is None or self.backbone_ckpt is None:
            raise ValueError("Run fit() first, or load bundle first.")
        if self._train_lib_sizes is None or self.keep_gene_idx is None or self.d_full is None or self.d is None:
            raise ValueError("Missing training stats (need fit() or load bundle).")

        dev = torch.device("cuda" if (self.device == "cuda" and torch.cuda.is_available()) else "cpu")

        from VAE.VAE_model import VAE
        vae = VAE(
            num_genes=int(self.d),
            device=str(dev),
            seed=self.seed,
            loss_ae="mse",
            hidden_dim=self.latent_dim,
            decoder_activation="ReLU",
        )
        vae.load_state_dict(torch.load(self.ae_ckpt, map_location="cpu"))
        vae.to(dev).eval()

        from guided_diffusion.script_util import create_model_and_diffusion, model_and_diffusion_defaults
        defaults = model_and_diffusion_defaults()
        defaults.update(dict(
            input_dim=self.latent_dim,
            hidden_dim=self.hidden_dim,
            diffusion_steps=self.diffusion_steps,
            noise_schedule=self.noise_schedule,
            class_cond=False,
        ))
        model, diffusion = create_model_and_diffusion(**defaults)
        model.load_state_dict(torch.load(self.backbone_ckpt, map_location="cpu"))
        model.to(dev).eval()

        n = int(n_samples)
        bs = int(batch_size)

        lat_list, got = [], 0
        while got < n:
            cur = min(bs, n - got)
            
            lat = diffusion.p_sample_loop(model,
                                          (cur, self.latent_dim), 
                                          clip_denoised=False, model_kwargs={})
            
            if isinstance(lat, tuple):
                lat = lat[0]
            elif isinstance(lat, dict) and "sample" in lat:
                lat = lat["sample"]
            
            lat_list.append(lat.detach().cpu().numpy())
            
            # lat = diffusion.p_sample_loop(model, (cur, self.latent_dim), clip_denoised=False, model_kwargs={})
            # lat_list.append(lat.detach().cpu().numpy())
            got += cur
        lat = np.concatenate(lat_list, axis=0)

        # decode to log1p-scale outputs, then expm1 back to relative expression
        x_log = vae(torch.tensor(lat, device=dev, dtype=torch.float32), return_decoded=True).detach().cpu().numpy()
        x_rel = np.expm1(np.maximum(x_log, 0.0)).astype(np.float64)
        probs = x_rel / np.maximum(x_rel.sum(axis=1, keepdims=True), 1e-12)

        rng = np.random.default_rng(self.seed)
        libs = self._train_lib_sizes
        libs = libs[libs > 0]
        if libs.size == 0:
            libs = np.array([1000], dtype=np.int64)

        Xf = np.empty((n, int(self.d)), dtype=np.int64)
        for i in range(n):
            L = int(rng.choice(libs))
            Xf[i] = rng.multinomial(L, probs[i])

        # pad back to full gene dimension
        Xfull = np.zeros((n, int(self.d_full)), dtype=np.int64)
        Xfull[:, self.keep_gene_idx] = Xf
        return Xfull
def save_scdiffusion_bundle(scdiff_model, bundle_dir):
    os.makedirs(bundle_dir, exist_ok=True)
    meta = dict(
        scd_root=scdiff_model.scd_root,
        workdir=scdiff_model.workdir,
        seed=scdiff_model.seed,
        ae_steps=scdiff_model.ae_steps,
        diff_steps=scdiff_model.diff_steps,
        batch_size=scdiff_model.batch_size,
        ae_ckpt_freq=scdiff_model.ae_ckpt_freq,
        diff_save_interval=scdiff_model.diff_save_interval,
        lr=scdiff_model.lr,
        weight_decay=scdiff_model.weight_decay,
        latent_dim=scdiff_model.latent_dim,
        hidden_dim=scdiff_model.hidden_dim,
        noise_schedule=scdiff_model.noise_schedule,
        diffusion_steps=scdiff_model.diffusion_steps,

        # new fields (gene filtering + padding)
        d_full=int(scdiff_model.d_full) if scdiff_model.d_full is not None else None,
        d_filtered=int(scdiff_model.d) if scdiff_model.d is not None else None,
        keep_gene_idx=(scdiff_model.keep_gene_idx.tolist() if scdiff_model.keep_gene_idx is not None else None),
        keep_cell_idx=(scdiff_model.keep_cell_idx.tolist() if scdiff_model.keep_cell_idx is not None else None),

        data_h5ad=scdiff_model.data_h5ad,
        ae_ckpt=scdiff_model.ae_ckpt,
        backbone_ckpt=scdiff_model.backbone_ckpt,
    )
    with open(os.path.join(bundle_dir, "scdiffusion_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    np.savez_compressed(
        os.path.join(bundle_dir, "scdiffusion_stats.npz"),
        train_lib_sizes=scdiff_model._train_lib_sizes.astype(np.int64),
    )
    print("Saved scDiffusion bundle:", bundle_dir)


def load_scdiffusion_bundle(bundle_dir, device="cuda"):
    meta = json.load(open(os.path.join(bundle_dir, "scdiffusion_meta.json"), "r"))
    stats = np.load(os.path.join(bundle_dir, "scdiffusion_stats.npz"))

    m = ScDiffusionModel(
        scd_root=meta["scd_root"],
        workdir=meta["workdir"],
        scimilarity_dir=None,  # not needed for sampling
        device=device,
        seed=meta["seed"],
        ae_steps=meta["ae_steps"],
        diff_steps=meta["diff_steps"],
        batch_size=meta["batch_size"],
        ae_ckpt_freq=meta["ae_ckpt_freq"],
        diff_save_interval=meta["diff_save_interval"],
        lr=meta["lr"],
        weight_decay=meta["weight_decay"],
        latent_dim=meta["latent_dim"],
        hidden_dim=meta["hidden_dim"],
        noise_schedule=meta["noise_schedule"],
        diffusion_steps=meta["diffusion_steps"],
    )

    # new fields
    m.d_full = meta.get("d_full", None)
    m.d = meta.get("d_filtered", meta.get("d", None))
    kg = meta.get("keep_gene_idx", None)
    kc = meta.get("keep_cell_idx", None)
    m.keep_gene_idx = None if kg is None else np.array(kg, dtype=int)
    m.keep_cell_idx = None if kc is None else np.array(kc, dtype=int)

    m.data_h5ad = meta["data_h5ad"]
    m.ae_ckpt = meta["ae_ckpt"]
    m.backbone_ckpt = meta["backbone_ckpt"]
    m._train_lib_sizes = stats["train_lib_sizes"].astype(np.int64)

    print("Loaded scDiffusion bundle:", bundle_dir)
    return m
