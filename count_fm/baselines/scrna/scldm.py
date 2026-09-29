"""Original SCLDMModel adapter; requires the upstream baseline environment."""
import os, json, pickle, subprocess
import numpy as np
import pandas as pd
import anndata as ad
from count_fm.experiments.base import UncondModel

class SCLDMModel(UncondModel):
    name = "scLDM"

    def _maybe_reset_run(self, exp_name, reset_if_broken=True):
        ckpt_dir = os.path.join(self.workdir, "experiments", "checkpoints", exp_name)
        last_ckpt = os.path.join(ckpt_dir, "last.ckpt")
        cfg_yaml  = os.path.join(ckpt_dir, "config.yaml")
    
        # if folder exists but missing the required files, it's a broken run
        if os.path.isdir(ckpt_dir) and reset_if_broken:
            ok = os.path.exists(last_ckpt) and os.path.exists(cfg_yaml)
            if not ok:
                print(f"[scLDM] broken run detected, deleting {ckpt_dir}")
                import shutil
                shutil.rmtree(ckpt_dir, ignore_errors=True)
        

    def __init__(
        self,
        scldm_root="scldm",                 # local clone of https://github.com/czi-ai/scLDM
        py_exec=None,                       # IMPORTANT: python in a py3.11+ env with scldm installed
        workdir="saved_models/scldm_dg",
        seed=0,
        device="cuda",
        vae_epochs=100,
        ldm_epochs=100,
        batch_size=16,
        devices=1,
        test_batch_size=32,
        num_workers=0,
        timesteps=50,
        guidance_weight=0.0,                # we set a dummy label with scale 0.0 for unconditional
    ):
        self.scldm_root = scldm_root
        self.py_exec = py_exec if py_exec is not None else "python"
        # self.workdir = workdir
        self.workdir = os.path.abspath(workdir)
        self.seed = int(seed)
        # self.device = str(device)
        self.devices = int(devices)

        self.vae_epochs = int(vae_epochs)
        self.ldm_epochs = int(ldm_epochs)
        self.batch_size = int(batch_size)
        self.test_batch_size = int(test_batch_size)
        self.num_workers = int(num_workers)
        self.timesteps = int(timesteps)
        self.guidance_weight = float(guidance_weight)

        self.data_dir = os.path.join(self.workdir, "data")
        self.infer_dir = os.path.join(self.workdir, "inference")
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.infer_dir, exist_ok=True)

        # fixed dummy conditioning to satisfy scLDM code path, but keep it unconditional via guidance_weight=0.0
        # self.label_key = "dummy"
        # self.label_vocab = {self.label_key: 1}            # one category: "0"
        # self.guidance_dict = {self.label_key: self.guidance_weight}

        self.label_key = "clusters"
        self.label_vocab = {self.label_key: 1}            # one category: "0"
        self.guidance_dict = {self.label_key: self.guidance_weight}
        

        # filled after fit
        self.d = None
        self.var_names = None
        self.train_h5ad = None
        self.test_h5ad = None
        self.mu_pkl = None
        self.sd_pkl = None
        self.vae_name = None
        self.ldm_name = None
        self.ckpt_root = None                             # base_experiment_path/checkpoints
        self.vae_ckpt_dir = None
        self.ldm_ckpt_dir = None

    def _assert_repo(self):
        if not os.path.isdir(self.scldm_root):
            raise FileNotFoundError(
                f"scldm_root not found: {self.scldm_root}\n"
                f"Clone scLDM repo there, and set SCLDM_PY to a py3.11 env with scldm installed."
            )


    def _run(self, cmd, cwd):
        print("[scLDM] running:")
        print(" ".join(cmd))
    
        env = os.environ.copy()
        env["WANDB_MODE"] = "disabled"
        env["WANDB_DISABLED"] = "true"
        env["WANDB_SILENT"] = "true"
    
        # IMPORTANT: do NOT force CUDA_VISIBLE_DEVICES here (don’t set it to "0")
        # Let your global setting decide (you’re on GPU 1).
        # (So: no env["CUDA_VISIBLE_DEVICES"]=... in this function.)
    
        # Wrap python script execution to force torch.load(weights_only=False)
        if len(cmd) >= 2 and cmd[1].endswith(".py"):
            script = cmd[1]
            args = cmd[2:]
    
            # one-liner, no indentation
            pycode = (
                "import sys,runpy,torch;"
                "_real=torch.load;"
                "torch.load=lambda *a,**kw:_real(*a,**dict(kw,weights_only=False));"
                f"sys.argv=[r'{script}']+sys.argv[1:];"
                f"runpy.run_path(r'{script}',run_name='__main__')"
            )
    
            cmd2 = [cmd[0], "-c", pycode] + args
            subprocess.run(cmd2, cwd=cwd, check=True, env=env)
        else:
            subprocess.run(cmd, cwd=cwd, check=True, env=env)


    def _write_train_test_h5ad(self, X_train_counts, X_test_counts, var_names):
        Xtr = np.asarray(X_train_counts)
        Xte = np.asarray(X_test_counts)

        if not np.issubdtype(Xtr.dtype, np.integer):
            Xtr = np.rint(Xtr).astype(np.int64)
        if not np.issubdtype(Xte.dtype, np.integer):
            Xte = np.rint(Xte).astype(np.int64)

        self.d = int(Xtr.shape[1])
        self.var_names = list(var_names) if var_names is not None else [str(i) for i in range(self.d)]

        # scLDM expects categorical labels in .obs for each class key
        # dummy_tr = pd.Categorical(np.full((Xtr.shape[0],), "0"))
        # dummy_te = pd.Categorical(np.full((Xte.shape[0],), "0"))

        # ad_tr = ad.AnnData(X=Xtr)
        # ad_tr.var_names = self.var_names
        # ad_tr.obs[self.label_key] = dummy_tr

        # ad_te = ad.AnnData(X=Xte)
        # ad_te.var_names = self.var_names
        # ad_te.obs[self.label_key] = dummy_te

        # self.train_h5ad = os.path.join(self.data_dir, "train.h5ad")
        # self.test_h5ad  = os.path.join(self.data_dir, "test.h5ad")


        lab_tr = pd.Categorical(np.full((Xtr.shape[0],), "0"))
        lab_te = pd.Categorical(np.full((Xte.shape[0],), "0"))
        
        ad_tr = ad.AnnData(X=Xtr)
        ad_tr.var_names = self.var_names
        ad_tr.obs[self.label_key] = lab_tr
        
        ad_te = ad.AnnData(X=Xte)
        ad_te.var_names = self.var_names
        ad_te.obs[self.label_key] = lab_te
        
        self.train_h5ad = os.path.abspath(os.path.join(self.data_dir, "train.h5ad"))
        self.test_h5ad  = os.path.abspath(os.path.join(self.data_dir, "test.h5ad"))


        ad_tr.write(self.train_h5ad)
        ad_te.write(self.test_h5ad)

        # size factor stats per label class (required if you set mu_size_factor / sd_size_factor)
        lib = Xtr.sum(axis=1).astype(np.float64)
        lib = np.maximum(lib, 1.0)
        log_sf = np.log(lib)
        mu = float(log_sf.mean())
        sd = float(log_sf.std(ddof=1) if log_sf.size > 1 else 1.0)

        mu_dict = {self.label_key: {"0": mu}}
        sd_dict = {self.label_key: {"0": sd}}

        self.mu_pkl = os.path.join(self.data_dir, "log_size_factor_mu.pkl")
        self.sd_pkl = os.path.join(self.data_dir, "log_size_factor_sd.pkl")
        with open(self.mu_pkl, "wb") as f:
            pickle.dump(mu_dict, f)
        with open(self.sd_pkl, "wb") as f:
            pickle.dump(sd_dict, f)

    def _common_overrides(self):
        # We override dataset_params.dentate_gyrus so scLDM uses our gene set and our dummy label.
        # We also force genes_seq_len = d so generated X has full gene dimension for your evaluation.
        d = int(self.d)
        return [
            f"seed={self.seed}",
            "datamodule.dataset=dentate_gyrus",
            f"datamodule.datamodule.train_adata_path={self.train_h5ad}",
            f"datamodule.datamodule.test_adata_path={self.test_h5ad}",
            f"datamodule.vocabulary_encoder.adata_path={self.train_h5ad}",
            "datamodule.dataset_params.dentate_gyrus.metadata_json=null",
            f"datamodule.dataset_params.dentate_gyrus.n_genes={d}",
            f"datamodule.dataset_params.dentate_gyrus.genes_seq_len={d}",
            "datamodule.dataset_params.dentate_gyrus.sample_genes=all",
            f"datamodule.dataset_params.dentate_gyrus.class_vocab_sizes.{self.label_key}=1",
            f"datamodule.dataset_params.dentate_gyrus.guidance_weight.{self.label_key}={self.guidance_weight}",
            # f"datamodule.dataset_params.dentate_gyrus.class_vocab_sizes={{{self.label_key}:1}}",
            # f"datamodule.dataset_params.dentate_gyrus.guidance_weight={{{self.label_key}:{self.guidance_weight}}}",
            f"datamodule.dataset_params.dentate_gyrus.mu_size_factor={self.mu_pkl}",
            f"datamodule.dataset_params.dentate_gyrus.sd_size_factor={self.sd_pkl}",
            f"datamodule.datamodule.batch_size={self.batch_size}",
            f"datamodule.datamodule.test_batch_size={self.test_batch_size}",
            f"datamodule.datamodule.num_workers={self.num_workers}",
            f"model.batch_size={self.batch_size}",
            f"model.test_batch_size={self.test_batch_size}",
        ]

    def fit(self, X_train_counts, X_test_counts, var_names=None, resume=True):
        self._assert_repo()
        os.makedirs(self.workdir, exist_ok=True)

        self._write_train_test_h5ad(X_train_counts, X_test_counts, var_names=var_names)

        self.vae_name = "vae_dg_custom"
        self.ldm_name = "ldm_dg_custom"

        self._maybe_reset_run(self.vae_name, reset_if_broken=True)
        self._maybe_reset_run(self.ldm_name, reset_if_broken=True)

        # where scLDM will place checkpoints/configs
        base_experiment_path = os.path.abspath(os.path.join(self.workdir, "experiments"))
        self.ckpt_root = os.path.join(base_experiment_path, "checkpoints")
        self.vae_ckpt_dir = os.path.join(self.ckpt_root, self.vae_name)
        self.ldm_ckpt_dir = os.path.join(self.ckpt_root, self.ldm_name)

        # 1) VAE training (auto resumes from last.ckpt if present)
        cmd_vae = [
            self.py_exec,
            os.path.join("experiments", "scripts", "train.py"),
            f"paths.base_data_path={os.path.abspath(self.data_dir)}",
            f"paths.base_experiment_path={base_experiment_path}",
            f"experiment_name={self.vae_name}",
            f"training.num_epochs={self.vae_epochs}",
            "training.trainer.enable_progress_bar=true",
        ] + self._common_overrides()

        self._run(cmd_vae, cwd=self.scldm_root)

        # 2) LDM training (auto resumes from last.ckpt if present)
        cmd_ldm = [
            self.py_exec,
            os.path.join("experiments", "scripts", "train_ldm.py"),
            f"paths.base_data_path={os.path.abspath(self.data_dir)}",
            f"paths.base_experiment_path={base_experiment_path}",
            f"experiment_name={self.ldm_name}",
            f"training.num_epochs={self.ldm_epochs}",
            "training.trainer.enable_progress_bar=true",
            f"model.module.vae_as_tokenizer.load_from_checkpoint.ckpt_path={self.ckpt_root}",
            f"model.module.vae_as_tokenizer.load_from_checkpoint.job_name={self.vae_name}",
            "model.module.vae_as_tokenizer.load_from_checkpoint.epoch=null",
        ] + self._common_overrides()

        self._run(cmd_ldm, cwd=self.scldm_root)

        print(f"[scLDM] trained. workdir={self.workdir}  resume={resume}")
        return self

    def _dup_keys(self, names):
        counts = {}
        out = []
        for n in names:
            k = counts.get(n, 0)
            out.append(f"{n}__dup{k}")
            counts[n] = k + 1
        return np.asarray(out, dtype=str)

    def _canonical_var_names(self):
        if not hasattr(self, "_canon_var_names") or self._canon_var_names is None:
            if self.train_h5ad is None or (not os.path.exists(self.train_h5ad)):
                raise RuntimeError("[scLDM] train_h5ad missing, cannot infer canonical var_names.")
            ad0 = ad.read_h5ad(self.train_h5ad)
            self._canon_var_names = np.asarray(ad0.var_names, dtype=str)
        return self._canon_var_names
        

    def _infer_one_round(
        self,
        gen_idx=0,
        timesteps=None,
        test_batch_size=None,
        guidance_weight=None,
        unconditional=True,
    ):
        # inference.py saves: {inference_path}/{dataset}_generated_{idx}.h5ad
        ckpt_file = os.path.join(self.ldm_ckpt_dir, "last.ckpt")
        cfg_file  = os.path.join(self.ldm_ckpt_dir, "config.yaml")
        if not os.path.exists(ckpt_file) or not os.path.exists(cfg_file):
            raise FileNotFoundError("Missing scLDM LDM checkpoint/config. Run fit() first or load bundle.")
    
        tb = self.test_batch_size if (test_batch_size is None) else int(test_batch_size)
        ts = self.timesteps if (timesteps is None) else int(timesteps)
        gw = self.guidance_weight if (guidance_weight is None) else float(guidance_weight)

        cmd_inf = [
            self.py_exec,
            os.path.join("experiments", "scripts", "inference.py"),
            f"ckpt_file={ckpt_file}",
            f"config_file={cfg_file}",
            f"inference_path={os.path.abspath(self.infer_dir)}",
            f"dataset_generation_idx={int(gen_idx)}",
        
            # what you control at generation time
            f"datamodule.datamodule.test_batch_size={tb}",
            f"++model.module.generation_args.timesteps={ts}",
            f"++model.module.generation_args.guidance_weight={{{self.label_key}:{gw}}}",
        
            # keep this to avoid unrelated interpolation crash
            "++datamodule.dataset_params.homo_sapiens.metadata_genes=null",
        ] + self._common_overrides()

        

        # cmd_inf = [
        #     self.py_exec,
        #     os.path.join("experiments", "scripts", "inference.py"),
        #     f"ckpt_file={ckpt_file}",
        #     f"config_file={cfg_file}",
        #     f"inference_path={os.path.abspath(self.infer_dir)}",
        #     f"dataset_generation_idx={int(gen_idx)}",
        
        #     # REQUIRED for predict dataloader
        #     f"++datamodule.datamodule.train_adata_path={self.train_h5ad}",
        #     f"++datamodule.datamodule.test_adata_path={self.test_h5ad}",
        
        #     # make encoder read the same gene set / categories
        #     f"++datamodule.vocabulary_encoder.adata_path={self.train_h5ad}",
        
        #     # your local size-factor pkls (avoid /work/_artifacts)
        #     f"++datamodule.dataset_params.dentate_gyrus.mu_size_factor={self.mu_pkl}",
        #     f"++datamodule.dataset_params.dentate_gyrus.sd_size_factor={self.sd_pkl}",
        
        #     # what you actually want to control at generation time
        #     f"datamodule.datamodule.test_batch_size={tb}",
        #     f"++model.module.generation_args.timesteps={ts}",
        #     f"++model.module.generation_args.guidance_weight={{{self.label_key}:{gw}}}",
        
        #     # avoid the interpolation crash path
        #     "++datamodule.dataset_params.homo_sapiens.metadata_genes=null",
        # ]
        
        self._run(cmd_inf, cwd=self.scldm_root)
    
        out_h5ad = os.path.join(self.infer_dir, f"dentate_gyrus_generated_{int(gen_idx)}.h5ad")
        if not os.path.exists(out_h5ad):
            raise FileNotFoundError("scLDM inference did not produce: " + out_h5ad)
    
        adata_gen = ad.read_h5ad(out_h5ad)
    
        # select unconditional or conditional half
        if "dataset" in adata_gen.obs:
            key = "generated_unconditional" if unconditional else "generated_conditional"
            ad_sel = adata_gen[adata_gen.obs["dataset"].values == key].copy()
            if ad_sel.n_obs == 0:
                # fallback: split by halves
                n = adata_gen.n_obs // 2
                ad_sel = adata_gen[:n].copy() if unconditional else adata_gen[n:].copy()
        else:
            n = adata_gen.n_obs // 2
            ad_sel = adata_gen[:n].copy() if unconditional else adata_gen[n:].copy()

        target = self._canonical_var_names()
        cur = np.asarray(ad_sel.var_names, dtype=str)
        
        if (len(cur) != len(target)) or (not np.array_equal(cur, target)):
            tkey = self._dup_keys(target)
            ckey = self._dup_keys(cur)
        
            pos = {k: i for i, k in enumerate(ckey)}
            idx = np.fromiter((pos.get(k, -1) for k in tkey), dtype=np.int64, count=len(tkey))
        
            if (idx < 0).any():
                missing = tkey[idx < 0][:10]
                raise RuntimeError(f"[scLDM] generated output missing columns (first 10 keys): {missing.tolist()}")
        
            ad_sel = ad_sel[:, idx].copy()
        
        Xg = ad_sel.X
        if hasattr(Xg, "toarray"):
            Xg = Xg.toarray()
        Xg = np.asarray(Xg)
        if not np.issubdtype(Xg.dtype, np.integer):
            Xg = np.rint(Xg).astype(np.int64)
        return Xg
    
    
    def sample(self, n_samples, n_steps=None, batch_size=None, unconditional=True):
        n = int(n_samples)
        ts = self.timesteps if (n_steps is None) else int(n_steps)
        tb = self.test_batch_size if (batch_size is None) else int(batch_size)
    
        chunks = []
        got = 0
        gen_idx = 0
        while got < n:
            Xg = self._infer_one_round(
                gen_idx=gen_idx,
                timesteps=ts,
                test_batch_size=tb,
                unconditional=bool(unconditional),
            )
            chunks.append(Xg)
            got += Xg.shape[0]
            gen_idx += 1
        X = np.concatenate(chunks, axis=0)[:n]
        return X
def save_scldm_bundle(scldm_model, bundle_dir):
    os.makedirs(bundle_dir, exist_ok=True)
    meta = dict(
        scldm_root=scldm_model.scldm_root,
        py_exec=scldm_model.py_exec,
        workdir=scldm_model.workdir,
        seed=scldm_model.seed,
        devices=getattr(scldm_model, "devices", 1),
        vae_epochs=scldm_model.vae_epochs,
        ldm_epochs=scldm_model.ldm_epochs,
        batch_size=scldm_model.batch_size,
        test_batch_size=scldm_model.test_batch_size,
        num_workers=scldm_model.num_workers,
        timesteps=scldm_model.timesteps,
        guidance_weight=scldm_model.guidance_weight,
        d=scldm_model.d,
        var_names=scldm_model.var_names,
        train_h5ad=scldm_model.train_h5ad,
        test_h5ad=scldm_model.test_h5ad,
        mu_pkl=scldm_model.mu_pkl,
        sd_pkl=scldm_model.sd_pkl,
        vae_name=scldm_model.vae_name,
        ldm_name=scldm_model.ldm_name,
        ckpt_root=scldm_model.ckpt_root,
    )
    with open(os.path.join(bundle_dir, "scldm_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print("Saved scLDM bundle:", bundle_dir)


def load_scldm_bundle(bundle_dir):
    meta = json.load(open(os.path.join(bundle_dir, "scldm_meta.json"), "r"))
    m = SCLDMModel(
        scldm_root=meta["scldm_root"],
        py_exec=meta["py_exec"],
        workdir=meta["workdir"],
        seed=meta["seed"],
        devices=meta.get("devices", 1),
        vae_epochs=meta["vae_epochs"],
        ldm_epochs=meta["ldm_epochs"],
        batch_size=meta["batch_size"],
        test_batch_size=meta["test_batch_size"],
        num_workers=meta["num_workers"],
        timesteps=meta["timesteps"],
        guidance_weight=meta["guidance_weight"],
    )
    m.d = meta.get("d", None)
    m.var_names = meta.get("var_names", None)
    m.train_h5ad = meta.get("train_h5ad", None)
    m.test_h5ad = meta.get("test_h5ad", None)
    m.mu_pkl = meta.get("mu_pkl", None)
    m.sd_pkl = meta.get("sd_pkl", None)
    m.vae_name = meta.get("vae_name", None)
    m.ldm_name = meta.get("ldm_name", None)
    m.ckpt_root = meta.get("ckpt_root", None)
    if m.ckpt_root is not None and m.vae_name is not None:
        base_experiment_path = os.path.abspath(os.path.join(m.workdir, "experiments"))
        m.ckpt_root = os.path.join(base_experiment_path, "checkpoints")
        m.vae_ckpt_dir = os.path.join(m.ckpt_root, m.vae_name)
        m.ldm_ckpt_dir = os.path.join(m.ckpt_root, m.ldm_name)
    print("Loaded scLDM bundle:", bundle_dir)
    return m
