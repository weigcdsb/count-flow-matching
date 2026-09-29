"""Original DCMModel adapter; requires the upstream baseline environment."""
import os, json, subprocess
from pathlib import Path
import numpy as np
import anndata as ad
import sys
from count_fm.experiments.base import UncondModel

class DCMModel(UncondModel):
    name = "DCM"

    def __init__(
        self,
        dcm_root="aivc-dcm",
        py_exec=None,                 # python>=3.10 recommended by the repo
        workdir="saved_models/dcm_dg",
        seed=0,
        device="cuda",
        # training
        hidden_dim=128,
        num_layers=4,
        num_heads=4,
        dropout=0.1,
        batch_size=32,
        num_epochs=50,
        learning_rate=1e-4,
        weight_decay=1e-2,
        mask_ratio=0.15,
        val_fraction=0.1,
        num_workers=0,
        save_interval=2,
        # sampling
        default_num_steps=50,
        default_temperature=1.0,
        # important: reserve a dedicated mask token (fixes repo's NUM_BINS=max(data) collision)
        apply_mask_token_fix=True,
    ):
        # self.dcm_root = str(dcm_root)
        self.dcm_root = str(Path(dcm_root).expanduser().resolve())
        self.py_exec = py_exec if py_exec is not None else sys.executable
        # self.workdir = str(workdir)
        self.workdir  = str(Path(workdir).expanduser().resolve())
        self.seed = int(seed)
        self.device = str(device)

        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.num_heads = int(num_heads)
        self.dropout = float(dropout)

        self.batch_size = int(batch_size)
        self.num_epochs = int(num_epochs)
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.mask_ratio = float(mask_ratio)
        self.val_fraction = float(val_fraction)
        self.num_workers = int(num_workers)
        self.save_interval = int(save_interval)

        self.default_num_steps = int(default_num_steps)
        self.default_temperature = float(default_temperature)

        self.apply_mask_token_fix = bool(apply_mask_token_fix)

        # filled after fit/load
        self.train_h5ad = None
        self.ckpt_dir = None
        self.mask_token = None   # integer token reserved for masking

    def _repo_path(self):
        p = Path(self.dcm_root)
        if not p.exists():
            raise FileNotFoundError(f"DCM repo not found at: {p}")
        return p

    def _scripts_dir(self):
        p = self._repo_path() / "scripts"
        if not p.exists():
            raise FileNotFoundError(f"DCM scripts/ not found under: {p}")
        return p

    def _subproc_env(self):
        env = os.environ.copy()
        repo = str(self._repo_path().resolve())
        src  = str((self._repo_path() / "src").resolve())
    
        parts = [src, repo]
        old = env.get("PYTHONPATH", "")
        if old:
            parts.append(old)
    
        env["PYTHONPATH"] = os.pathsep.join(parts)
        env["PYTORCH_NVML_BASED_CUDA_CHECK"] = "0"
        env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        return env

    def _assert_cuda_in_subproc(self):
        cmd = [self.py_exec, "-c", "import torch; print(int(torch.cuda.is_available()), torch.cuda.device_count())"]
        r = subprocess.run(cmd, env=self._subproc_env(), capture_output=True, text=True)
        out = (r.stdout + "\n" + r.stderr).strip()
        print("DCM cuda check:", out)
        if r.returncode != 0 or out.startswith("0 "):
            raise RuntimeError("DCM subprocess has no CUDA. Run this notebook on a GPU node (nvidia-smi must work) or use a CUDA-enabled torch env.")        

        

    def _ensure_maskfix_scripts(self):
        """
        The upstream scripts set:
            NUM_BINS = int(dataset.max().item())
            VOCAB_SIZE = NUM_BINS + 1  # +1 for mask token
        But model code uses mask_index=num_bins, so mask token collides with max count.
        Fix: NUM_BINS = max_count + 1, so mask token is a dedicated extra state.
        We do this by writing patched copies inside repo/scripts so sys.path logic still works.
        """
        scripts = self._scripts_dir()
        train_src = scripts / "train_rnaseq.py"
        infer_src = scripts / "infernce_generation.py"  # repo spelling

        train_dst = scripts / "train_rnaseq_maskfix.py"
        infer_dst = scripts / "infernce_generation_maskfix.py"

        def _patch_one(src, dst):
            if dst.exists():
                return str(dst)
            txt = src.read_text()
            needle = "NUM_BINS = int(dataset.max().item())"
            if needle not in txt:
                raise RuntimeError(f"Expected pattern not found in {src.name}: {needle}")
            txt2 = txt.replace(needle, "NUM_BINS = int(dataset.max().item()) + 1", 1)
            dst.write_text(txt2)
            return str(dst)

        train_script = str(train_src)
        infer_script = str(infer_src)
        if self.apply_mask_token_fix:
            train_script = _patch_one(train_src, train_dst)
            infer_script = _patch_one(infer_src, infer_dst)

        # return train_script, infer_script
        return str(Path(train_script).resolve()), str(Path(infer_script).resolve())

    def _write_train_h5ad(self, X_train_counts, var_names=None):
        Path(self.workdir).mkdir(parents=True, exist_ok=True)
        out = Path(self.workdir) / "dcm_train_counts.h5ad"

        X = np.asarray(X_train_counts)
        if not np.issubdtype(X.dtype, np.integer):
            X = np.rint(X).astype(np.int64)

        # define the dedicated mask token we want the model to use
        self.mask_token = int(X.max()) + 1

        adata_tmp = ad.AnnData(X.astype(np.int32, copy=False))
        if var_names is not None:
            adata_tmp.var_names = np.asarray(var_names, dtype=str)

        # store metadata (not used by scripts, but useful for debugging)
        adata_tmp.uns["dcm_mask_token"] = int(self.mask_token)
        adata_tmp.uns["dcm_max_count"] = int(X.max())

        adata_tmp.write_h5ad(out)
        return str(out)

    def fit(self, X_train_counts, var_names=None, resume=True, **kwargs):
        # allow overrides
        num_epochs = int(kwargs.get("num_epochs", self.num_epochs))
        batch_size = int(kwargs.get("batch_size", self.batch_size))
        learning_rate = float(kwargs.get("learning_rate", self.learning_rate))
        weight_decay = float(kwargs.get("weight_decay", self.weight_decay))
        mask_ratio = float(kwargs.get("mask_ratio", self.mask_ratio))
        val_fraction = float(kwargs.get("val_fraction", self.val_fraction))

        self.train_h5ad = self._write_train_h5ad(X_train_counts, var_names=var_names)
        # self.ckpt_dir = str(Path(self.workdir) / "checkpoints")
        self.ckpt_dir = str((Path(self.workdir) / "checkpoints").resolve())
        Path(self.ckpt_dir).mkdir(parents=True, exist_ok=True)
        config_path = str((self._repo_path() / "configs" / "rnaseq_small.yaml").resolve())


        train_script, _ = self._ensure_maskfix_scripts()

        self._assert_cuda_in_subproc()
        cmd = [
            self.py_exec, train_script,
            # "--config", str(Path(self._repo_path()) / "configs" / "rnaseq_small.yaml"),
            # "--data_path", self.train_h5ad,
            # "--checkpoint_dir", self.ckpt_dir,

            "--config", config_path,
            "--data_path", str(Path(self.train_h5ad).resolve()),
            "--checkpoint_dir", self.ckpt_dir,
            
            "--seed", str(self.seed),

            "--hidden_dim", str(self.hidden_dim),
            "--num_layers", str(self.num_layers),
            "--num_heads", str(self.num_heads),
            "--dropout", str(self.dropout),

            "--batch_size", str(batch_size),
            "--num_epochs", str(num_epochs),
            "--learning_rate", str(learning_rate),
            "--weight_decay", str(weight_decay),
            "--mask_ratio", str(mask_ratio),
            "--val_fraction", str(val_fraction),
            "--num_workers", str(self.num_workers),
            "--save_interval", str(self.save_interval),
        ]
        if resume:
            cmd += ["--resume", "auto"]

        print("DCM train cmd:")
        print(" ".join(cmd))
        # subprocess.run(cmd, cwd=str(self._repo_path()), check=True)
        subprocess.run(cmd, cwd=str(self._repo_path()), check=True, env=self._subproc_env())
        print("DCM training done. ckpt_dir =", self.ckpt_dir)


    def sample(self, n_samples, n_steps=None, temperature=None, gen_chunk=16, **kwargs):
        if self.ckpt_dir is None or self.train_h5ad is None:
            raise RuntimeError("DCMModel.sample called before fit/load.")
    
        n_samples = int(n_samples)
        gen_chunk = int(gen_chunk)
        if gen_chunk <= 0:
            raise ValueError("gen_chunk must be positive")
    
        n_steps = self.default_num_steps if n_steps is None else int(n_steps)
        temperature = self.default_temperature if temperature is None else float(temperature)
    
        _, infer_script = self._ensure_maskfix_scripts()
    
        # helps fragmentation a bit, harmless otherwise
        env = self._subproc_env()
        env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    
        outs = []
        done = 0
        chunk_id = 0
    
        while done < n_samples:
            cur = min(gen_chunk, n_samples - done)
    
            cmd = [
                self.py_exec, infer_script,
                "--experiment_dir", str(Path(self.ckpt_dir).resolve()),
                "--data_path", str(Path(self.train_h5ad).resolve()),
                "--num_generate", str(cur),
                "--num_steps", str(n_steps),
                "--temperature", str(temperature),
                "--seed", str(int(self.seed) + chunk_id),   # avoid identical chunks
                "--val_fraction", str(self.val_fraction),
                "--use_train_split",
            ]
    
            print(f"DCM sample cmd (chunk {chunk_id}, n={cur}):")
            print(" ".join(cmd))
    
            subprocess.run(cmd, cwd=str(self._repo_path()), check=True, env=env)
    
            out_dir = Path(self.ckpt_dir) / "generation_results"
            npy_path = out_dir / "generated_cells.npy"
            if not npy_path.exists():
                raise FileNotFoundError(f"DCM generation output not found: {npy_path}")
    
            Xg = np.load(npy_path)
            Xg = np.asarray(Xg, dtype=np.int64)
            outs.append(Xg)
    
            done += cur
            chunk_id += 1
    
        X = np.concatenate(outs, axis=0)
    
        # map any leftover mask tokens to 0 (safe)
        if self.mask_token is None:
            self.mask_token = int(np.max(X))
        X[X == int(self.mask_token)] = 0
    
        return X
def save_dcm_bundle(dcm_model, bundle_dir):
    os.makedirs(bundle_dir, exist_ok=True)
    meta = dict(
        dcm_root=dcm_model.dcm_root,
        py_exec=dcm_model.py_exec,
        workdir=dcm_model.workdir,
        seed=dcm_model.seed,
        device=dcm_model.device,

        hidden_dim=dcm_model.hidden_dim,
        num_layers=dcm_model.num_layers,
        num_heads=dcm_model.num_heads,
        dropout=dcm_model.dropout,

        batch_size=dcm_model.batch_size,
        num_epochs=dcm_model.num_epochs,
        learning_rate=dcm_model.learning_rate,
        weight_decay=dcm_model.weight_decay,
        mask_ratio=dcm_model.mask_ratio,
        val_fraction=dcm_model.val_fraction,
        num_workers=dcm_model.num_workers,
        save_interval=dcm_model.save_interval,

        default_num_steps=dcm_model.default_num_steps,
        default_temperature=dcm_model.default_temperature,

        apply_mask_token_fix=dcm_model.apply_mask_token_fix,

        train_h5ad=dcm_model.train_h5ad,
        ckpt_dir=dcm_model.ckpt_dir,
        mask_token=(int(dcm_model.mask_token) if dcm_model.mask_token is not None else None),
    )
    with open(os.path.join(bundle_dir, "dcm_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print("Saved DCM bundle:", bundle_dir)


def load_dcm_bundle(bundle_dir):
    meta = json.load(open(os.path.join(bundle_dir, "dcm_meta.json"), "r"))
    m = DCMModel(
        dcm_root=meta["dcm_root"],
        py_exec=meta["py_exec"],
        workdir=meta["workdir"],
        seed=meta["seed"],
        device=meta["device"],

        hidden_dim=meta["hidden_dim"],
        num_layers=meta["num_layers"],
        num_heads=meta["num_heads"],
        dropout=meta["dropout"],

        batch_size=meta["batch_size"],
        num_epochs=meta["num_epochs"],
        learning_rate=meta["learning_rate"],
        weight_decay=meta["weight_decay"],
        mask_ratio=meta["mask_ratio"],
        val_fraction=meta["val_fraction"],
        num_workers=meta["num_workers"],
        save_interval=meta["save_interval"],

        default_num_steps=meta["default_num_steps"],
        default_temperature=meta["default_temperature"],
        apply_mask_token_fix=meta.get("apply_mask_token_fix", True),
    )
    m.train_h5ad = meta.get("train_h5ad", None)
    m.ckpt_dir   = meta.get("ckpt_dir", None)
    m.mask_token = meta.get("mask_token", None)
    return m
