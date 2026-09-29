"""Original CFGenModel adapter; requires the upstream baseline environment."""
import os, json, math
import numpy as np
import pandas as pd
import torch
try:
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import ModelCheckpoint
except Exception:
    import lightning.pytorch as pl
    from lightning.pytorch.callbacks import ModelCheckpoint
from anndata import AnnData
from cfgen.data.scrnaseq_loader import RNAseqLoader
from cfgen.models.base.encoder_model import EncoderModel
from cfgen.models.featurizers.category_featurizer import CategoricalFeaturizer
from cfgen.models.fm.denoising_model import MLPTimeStep
from cfgen.models.fm.fm import FM
import re
from count_fm.experiments.base import UncondModel

def _torch_load_trusted(path, map_location="cpu"):
    # For your own checkpoints, allow full unpickling.
    # This avoids PyTorch 2.6 default weights_only=True failures.
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        # older torch that doesn't have weights_only
        return torch.load(path, map_location=map_location)


class CFGenModel(UncondModel):
    name = "CFGen"

    def __init__(
        self,
        device="cuda",
        workdir="saved_models/cfgen_dg_v1",
        seed=0,

        # We train "unconditional" by using a constant covariate with 1 category.
        covariate_key="dummy",

        # Dataset / loader
        layer_key="X_counts",
        normalization_type="log_gexp",
        subsample_frac=1.0,
        split_rates=(0.90, 0.10),

        # Encoder (AE) config (matches CFGen defaults)
        encoder_dims=(512, 256, 50),
        encoder_lr=1e-3,
        encoder_wd=1e-5,
        encoder_max_epochs=300,
        encoder_batch_size=256,

        # Flow (FM) config (matches CFGen defaults)
        fm_hidden_dim=32,
        fm_dropout_prob=0.0,
        fm_n_blocks=3,
        fm_embedding_dim=20,
        fm_conditional=False,          # IMPORTANT: unconditional training
        fm_normalization="none",
        fm_embed_size_factor=False,
        fm_conditioning_probability=0.8,
        fm_guided_conditioning=True,

        fm_lr=1e-4,
        fm_wd=1e-6,
        fm_sigma=1e-4,
        fm_use_ot=False,
        fm_antithetic_time_sampling=True,
        fm_max_epochs=1500,
        fm_batch_size=256,

        # Feature embedding
        one_hot_encode_features=False,
    ):
        self.device = device
        self.workdir = workdir
        self.seed = int(seed)

        self.covariate_key = str(covariate_key)
        self.layer_key = str(layer_key)
        self.normalization_type = str(normalization_type)
        self.subsample_frac = float(subsample_frac)
        self.split_rates = tuple(split_rates)

        self.encoder_dims = tuple(int(x) for x in encoder_dims)
        self.encoder_lr = float(encoder_lr)
        self.encoder_wd = float(encoder_wd)
        self.encoder_max_epochs = int(encoder_max_epochs)
        self.encoder_batch_size = int(encoder_batch_size)

        self.fm_hidden_dim = int(fm_hidden_dim)
        self.fm_dropout_prob = float(fm_dropout_prob)
        self.fm_n_blocks = int(fm_n_blocks)
        self.fm_embedding_dim = int(fm_embedding_dim)
        self.fm_conditional = bool(fm_conditional)
        self.fm_normalization = str(fm_normalization)
        self.fm_embed_size_factor = bool(fm_embed_size_factor)
        self.fm_conditioning_probability = float(fm_conditioning_probability)
        self.fm_guided_conditioning = bool(fm_guided_conditioning)

        self.fm_lr = float(fm_lr)
        self.fm_wd = float(fm_wd)
        self.fm_sigma = float(fm_sigma)
        self.fm_use_ot = bool(fm_use_ot)
        self.fm_antithetic_time_sampling = bool(fm_antithetic_time_sampling)
        self.fm_max_epochs = int(fm_max_epochs)
        self.fm_batch_size = int(fm_batch_size)

        self.one_hot_encode_features = bool(one_hot_encode_features)

        # trained artifacts
        self.encoder_model = None
        self.fm_model = None
        self._meta = None  # for save/load

        os.makedirs(self.workdir, exist_ok=True)

    def _dev(self):
        if self.device == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def _make_train_adata(self, X_train_counts, var_names=None):
        X = np.asarray(X_train_counts)
        if not np.issubdtype(X.dtype, np.integer):
            X = np.rint(X).astype(np.int64)

        adata = AnnData(X=X)
        if var_names is not None:
            adata.var_names = list(var_names)

        # constant covariate (1 category)
        adata.obs[self.covariate_key] = pd.Categorical(np.zeros((X.shape[0],), dtype=int))

        # CFGen loader expects a layer_key; if absent, it copies from X anyway, but keep explicit.
        adata.layers[self.layer_key] = adata.X.copy()
        return adata

    def fit(self, X_train_counts, var_names=None, resume=True):
        dev = self._dev()
        pl.seed_everything(self.seed, workers=True)

        # build AnnData in-memory (no muon read path needed)
        adata_tr = self._make_train_adata(X_train_counts, var_names=var_names)

        # dataset / loader
        ds = RNAseqLoader(
            adata_tr,
            layer_key=self.layer_key,
            covariate_keys=[self.covariate_key],
            subsample_frac=self.subsample_frac,
            normalization_type=self.normalization_type,
            is_binarized=False,
        )

        # split
        n = len(ds)
        n_tr = int(round(self.split_rates[0] * n))
        n_va = n - n_tr
        g = torch.Generator().manual_seed(self.seed)
        train_data, valid_data = torch.utils.data.random_split(ds, [n_tr, n_va], generator=g)

        dl_tr_enc = torch.utils.data.DataLoader(
            train_data, batch_size=self.encoder_batch_size, shuffle=True, num_workers=0, drop_last=True
        )
        dl_va_enc = torch.utils.data.DataLoader(
            valid_data, batch_size=self.encoder_batch_size, shuffle=False, num_workers=0, drop_last=True
        )

        # -------------------------
        # (1) Train encoder
        # -------------------------
        enc_ckpt_dir = os.path.join(self.workdir, "cfgen_encoder_ckpts")
        os.makedirs(enc_ckpt_dir, exist_ok=True)
        enc_last = os.path.join(enc_ckpt_dir, "last.ckpt")

        encoder_kwargs = {
            "rna": {
                "dims": list(self.encoder_dims),
                "batch_norm": True,
                "dropout": False,
                "dropout_p": 0.0,
            }
        }

        enc = EncoderModel(
            in_dim={"rna": ds.X["rna"].shape[1]},
            encoder_kwargs=encoder_kwargs,
            learning_rate=self.encoder_lr,
            weight_decay=self.encoder_wd,
            covariate_specific_theta=False,
            conditioning_covariate=self.covariate_key,
            n_cat=None,
            is_binarized=False,
        )

        enc_ckpt_cb = ModelCheckpoint(
            dirpath=enc_ckpt_dir,
            filename="epoch_{epoch:03d}",
            monitor="valid/loss",
            mode="min",
            every_n_epochs=20,
            save_last=True,
            auto_insert_metric_name=False,
        )

        enc_trainer = pl.Trainer(
            max_epochs=self.encoder_max_epochs,
            accelerator="gpu" if (dev.type == "cuda") else "cpu",
            devices=1,
            logger=False,
            callbacks=[enc_ckpt_cb],
            enable_checkpointing=True,
            log_every_n_steps=50,
        )

        ckpt_path = enc_last if (resume and os.path.exists(enc_last)) else None
        enc_trainer.fit(enc, train_dataloaders=dl_tr_enc, val_dataloaders=dl_va_enc, ckpt_path=ckpt_path)

        # freeze encoder for FM
        for p in enc.parameters():
            p.requires_grad = False
        enc.eval()

        # -------------------------
        # (2) Train FM in latent
        # -------------------------
        dl_tr_fm = torch.utils.data.DataLoader(
            train_data, batch_size=self.fm_batch_size, shuffle=True, num_workers=0, drop_last=True
        )
        dl_va_fm = torch.utils.data.DataLoader(
            valid_data, batch_size=self.fm_batch_size, shuffle=False, num_workers=0, drop_last=True
        )

        # categorical featurizer (still needed by CFGen code path)
        n_cat = len(ds.id2cov[self.covariate_key])
        feat = CategoricalFeaturizer(
            n_cat=n_cat,
            one_hot_encode_features=self.one_hot_encode_features,
            device=dev,
            embedding_dimensions=self.fm_embedding_dim,
        )

        feature_embeddings = {self.covariate_key: feat}

        denoise = MLPTimeStep(
            in_dim=self.encoder_dims[-1],
            hidden_dim=self.fm_hidden_dim,
            dropout_prob=self.fm_dropout_prob,
            n_blocks=self.fm_n_blocks,
            size_factor_min=ds.min_size_factor,
            size_factor_max=ds.max_size_factor,
            embed_size_factor=self.fm_embed_size_factor,
            covariate_list=[self.covariate_key],
            embedding_dim=self.fm_embedding_dim,
            normalization=self.fm_normalization,
            conditional=self.fm_conditional,
            is_binarized=False,
            modality_list=["rna"],
            conditioning_probability=self.fm_conditioning_probability,
            guided_conditioning=self.fm_guided_conditioning,
        ).to(dev)

        fm = FM(
            encoder_model=enc,
            denoising_model=denoise,
            feature_embeddings=feature_embeddings,
            plotting_folder=None,  # disable CFGen internal plotting
            in_dim={"rna": self.encoder_dims[-1]},
            size_factor_statistics={"mean": ds.log_size_factor_mu, "sd": ds.log_size_factor_sd},
            covariate_list=[self.covariate_key],
            theta_covariate=self.covariate_key,
            size_factor_covariate=self.covariate_key,
            encoder_type="fixed",
            learning_rate=self.fm_lr,
            weight_decay=self.fm_wd,
            antithetic_time_sampling=self.fm_antithetic_time_sampling,
            sigma=self.fm_sigma,
            covariate_specific_theta=False,
            plot_and_eval_every=10**9,  # effectively never
            use_ot=self.fm_use_ot,
            is_binarized=False,
            modality_list=["rna"],
            guidance_weights={self.covariate_key: 1},
        )

        fm_ckpt_dir = os.path.join(self.workdir, "cfgen_fm_ckpts")
        os.makedirs(fm_ckpt_dir, exist_ok=True)
        fm_last = os.path.join(fm_ckpt_dir, "last.ckpt")

        fm_ckpt_cb = ModelCheckpoint(
            dirpath=fm_ckpt_dir,
            filename="epoch_{epoch:03d}",
            monitor="valid/loss",
            mode="min",
            every_n_epochs=50,
            save_last=True,
            auto_insert_metric_name=False,
        )

        fm_trainer = pl.Trainer(
            max_epochs=self.fm_max_epochs,
            accelerator="gpu" if (dev.type == "cuda") else "cpu",
            devices=1,
            logger=False,
            callbacks=[fm_ckpt_cb],
            enable_checkpointing=True,
            log_every_n_steps=50,
            gradient_clip_val=1.0,
        )

        ckpt_path = fm_last if (resume and os.path.exists(fm_last)) else None
        fm_trainer.fit(fm, train_dataloaders=dl_tr_fm, val_dataloaders=dl_va_fm, ckpt_path=ckpt_path)

        fm.eval()

        # stash
        self.encoder_model = enc
        self.fm_model = fm

        self._meta = dict(
            d=int(ds.X["rna"].shape[1]),
            covariate_key=self.covariate_key,
            layer_key=self.layer_key,
            normalization_type=self.normalization_type,
            encoder_kwargs=encoder_kwargs,
            encoder_dims=self.encoder_dims,
            fm_params=dict(
                hidden_dim=self.fm_hidden_dim,
                dropout_prob=self.fm_dropout_prob,
                n_blocks=self.fm_n_blocks,
                embedding_dim=self.fm_embedding_dim,
                conditional=self.fm_conditional,
                normalization=self.fm_normalization,
                embed_size_factor=self.fm_embed_size_factor,
                conditioning_probability=self.fm_conditioning_probability,
                guided_conditioning=self.fm_guided_conditioning,
                lr=self.fm_lr,
                wd=self.fm_wd,
                sigma=self.fm_sigma,
                use_ot=self.fm_use_ot,
                antithetic_time_sampling=self.fm_antithetic_time_sampling,
                one_hot_encode_features=self.one_hot_encode_features,
            ),
            id2cov={self.covariate_key: list(ds.id2cov[self.covariate_key].keys())},
            size_factor_statistics=dict(mean=ds.log_size_factor_mu, sd=ds.log_size_factor_sd),
            min_size_factor=ds.min_size_factor,
            max_size_factor=ds.max_size_factor,
        )

        print(f"CFGen trained. workdir={self.workdir}  resume={resume}  device={dev}")
        return self

    @torch.no_grad()
    def sample(self, n_samples, n_steps=50, batch_size=512, unconditional=True):
        assert self.fm_model is not None, "Call fit() or load_cfgen_bundle() first."
        dev = self._dev()

        n = int(n_samples)
        bs = int(batch_size)
        reps = int(math.ceil(n / bs))

        # covariate_indices: constant 0 (since dummy covariate has 1 class)
        # covariate_indices = {self.covariate_key: torch.zeros((reps * bs,), dtype=torch.long)}
        covariate_indices = {self.covariate_key: torch.zeros((reps * bs,), dtype=torch.long, device=dev)}


        # --- make size_factor_statistics indexable (CFGen expects 1D tensors) ---
        sf = self.fm_model.size_factor_statistics
        
        def _as_1d_tensor(x):
            if torch.is_tensor(x):
                t = x.to(dev)
            else:
                t = torch.tensor(x, device=dev)
            if t.ndim == 0:
                t = t.reshape(1)
            return t
        
        for mod in self.fm_model.modality_list:
            for k in ("mean", "sd"):
                # sf[k] might be a float (bad), dict(mod->...), or dict(mod->dict(cov->...))
                if not isinstance(sf.get(k, None), dict):
                    sf[k] = {mod: {self.covariate_key: sf[k]}}
        
                if mod not in sf[k]:
                    # fallback if keys differ
                    sf[k][mod] = sf[k].get("rna", next(iter(sf[k].values())))
        
                if not isinstance(sf[k][mod], dict):
                    sf[k][mod] = {self.covariate_key: sf[k][mod]}
        
                if self.covariate_key not in sf[k][mod]:
                    sf[k][mod][self.covariate_key] = next(iter(sf[k][mod].values()))
        
                sf[k][mod][self.covariate_key] = _as_1d_tensor(sf[k][mod][self.covariate_key])
        


        
        out = self.fm_model.batched_sample(
            batch_size=bs,
            repetitions=reps,
            n_sample_steps=int(n_steps),
            theta_covariate=self.covariate_key,
            size_factor_covariate=self.covariate_key,
            conditioning_covariates=[self.covariate_key],
            covariate_indices=covariate_indices,
            log_size_factor=None,
            unconditional=bool(unconditional),
        )["rna"]

        X = out.cpu().numpy()[:n]
        X = np.asarray(X)
        if not np.issubdtype(X.dtype, np.integer):
            X = np.rint(X).astype(np.int64)
        return X
def _jsonify(x):
    if isinstance(x, dict):
        return {str(k): _jsonify(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonify(v) for v in x]
    if isinstance(x, (set,)):
        return [_jsonify(v) for v in sorted(list(x))]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    if torch.is_tensor(x):
        return x.detach().cpu().tolist()
    return x


def _torch_load_weights(path, map_location="cpu"):
    # Works on torch<2.6 and torch>=2.6 (weights-only tensors)
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _infer_encoder_dims_and_d_from_sd(sd):
    pat = re.compile(r"^encoder\.rna\.net\.(\d+)(?:\.(\d+))?\.weight$")
    layers = {}
    for k, v in sd.items():
        m = pat.match(k)
        if m and isinstance(v, torch.Tensor) and v.ndim == 2:
            i = int(m.group(1))
            layers.setdefault(i, (k, v))
    idx = sorted(layers.keys())
    if not idx:
        raise RuntimeError("Cannot infer encoder dims from encoder state_dict keys.")
    d = int(layers[idx[0]][1].shape[1])
    dims = [int(layers[i][1].shape[0]) for i in idx]
    return d, dims


def _remap_keys_to_match_target(src_sd, tgt_keys):
    out = {}
    for k, v in src_sd.items():
        if k in tgt_keys:
            out[k] = v
            continue
        k2 = re.sub(r"\.net\.(\d+)\.weight$", r".net.\1.0.weight", k)
        if k2 != k and k2 in tgt_keys:
            out[k2] = v
            continue
        k2 = re.sub(r"\.net\.(\d+)\.bias$", r".net.\1.0.bias", k)
        if k2 != k and k2 in tgt_keys:
            out[k2] = v
            continue
        k2 = re.sub(r"\.net\.(\d+)\.0\.weight$", r".net.\1.weight", k)
        if k2 != k and k2 in tgt_keys:
            out[k2] = v
            continue
        k2 = re.sub(r"\.net\.(\d+)\.0\.bias$", r".net.\1.bias", k)
        if k2 != k and k2 in tgt_keys:
            out[k2] = v
            continue
    return out


def save_cfgen_bundle(cfgen_model, bundle_dir):
    # IMPORTANT: save from in-memory models (NO loading .ckpt -> no pickle/safe_globals issues)
    assert cfgen_model.encoder_model is not None and cfgen_model.fm_model is not None, \
        "CFGen not trained/loaded in memory: encoder_model or fm_model is None."
    assert cfgen_model._meta is not None, "CFGen has no _meta. Call fit() first."

    os.makedirs(bundle_dir, exist_ok=True)

    torch.save(cfgen_model.encoder_model.state_dict(),
               os.path.join(bundle_dir, "cfgen_encoder_sd.pt"))
    torch.save(cfgen_model.fm_model.state_dict(),
               os.path.join(bundle_dir, "cfgen_fm_sd.pt"))

    with open(os.path.join(bundle_dir, "cfgen_meta.json"), "w") as f:
        json.dump(_jsonify(cfgen_model._meta), f, indent=2)

    print("Saved CFGen bundle (weights-only):", bundle_dir)


def load_cfgen_bundle(bundle_dir, device="cuda"):
    meta = json.load(open(os.path.join(bundle_dir, "cfgen_meta.json"), "r"))
    enc_path = os.path.join(bundle_dir, "cfgen_encoder_sd.pt")
    fm_path  = os.path.join(bundle_dir, "cfgen_fm_sd.pt")

    dev = torch.device("cuda" if (device == "cuda" and torch.cuda.is_available()) else "cpu")

    def _as_float_rna(v):
        while True:
            if isinstance(v, dict):
                v = v["rna"] if ("rna" in v) else next(iter(v.values()))
                continue
            if isinstance(v, (list, tuple, np.ndarray)):
                v = v[0]
                continue
            if torch.is_tensor(v):
                v = v.detach().cpu().flatten()[0].item()
                continue
            return float(v)

    # ---------- load weights-only ----------
    enc_sd = _torch_load_weights(enc_path, map_location="cpu")
    fm_sd  = _torch_load_weights(fm_path,  map_location="cpu")

    # ---------- rebuild encoder from its state_dict ----------
    d_ckpt, dims_ckpt = _infer_encoder_dims_and_d_from_sd(enc_sd)
    zdim = int(dims_ckpt[-1])
    has_bn = any(".running_mean" in k for k in enc_sd.keys()) or any(".running_var" in k for k in enc_sd.keys())

    encoder_kwargs = {"rna": {"dims": list(dims_ckpt), "batch_norm": bool(has_bn), "dropout": False, "dropout_p": 0.0}}
    enc = EncoderModel(
        in_dim={"rna": d_ckpt},
        encoder_kwargs=encoder_kwargs,
        learning_rate=1e-3,
        weight_decay=1e-5,
        covariate_specific_theta=False,
        conditioning_covariate=meta["covariate_key"],
        n_cat=None,
        is_binarized=False,
    )

    tgt_keys = set(enc.state_dict().keys())
    enc_sd2 = _remap_keys_to_match_target(enc_sd, tgt_keys)
    enc.load_state_dict(enc_sd2, strict=False)
    enc.eval()
    for p in enc.parameters():
        p.requires_grad = False
    enc = enc.to(dev)

    # ---------- rebuild FM ----------
    cov = meta["covariate_key"]
    fm_params = meta["fm_params"]
    n_cat = len(meta["id2cov"][cov])

    feat = CategoricalFeaturizer(
        n_cat=n_cat,
        one_hot_encode_features=bool(fm_params["one_hot_encode_features"]),
        device=dev,
        embedding_dimensions=int(fm_params["embedding_dim"]),
    )
    feature_embeddings = {cov: feat}

    denoise = MLPTimeStep(
        in_dim=zdim,
        hidden_dim=int(fm_params["hidden_dim"]),
        dropout_prob=float(fm_params["dropout_prob"]),
        n_blocks=int(fm_params["n_blocks"]),
        size_factor_min=_as_float_rna(meta["min_size_factor"]),
        size_factor_max=_as_float_rna(meta["max_size_factor"]),
        embed_size_factor=bool(fm_params["embed_size_factor"]),
        covariate_list=[cov],
        embedding_dim=int(fm_params["embedding_dim"]),
        normalization=str(fm_params["normalization"]),
        conditional=bool(fm_params["conditional"]),
        is_binarized=False,
        modality_list=["rna"],
        conditioning_probability=float(fm_params["conditioning_probability"]),
        guided_conditioning=bool(fm_params["guided_conditioning"]),
    ).to(dev)

    sf = meta["size_factor_statistics"]
    fm = FM(
        encoder_model=enc,
        denoising_model=denoise,
        feature_embeddings=feature_embeddings,
        plotting_folder=None,
        in_dim={"rna": zdim},
        size_factor_statistics={"mean": _as_float_rna(sf["mean"]), "sd": _as_float_rna(sf["sd"])},
        covariate_list=[cov],
        theta_covariate=cov,
        size_factor_covariate=cov,
        encoder_type="fixed",
        learning_rate=float(fm_params["lr"]),
        weight_decay=float(fm_params["wd"]),
        antithetic_time_sampling=bool(fm_params["antithetic_time_sampling"]),
        sigma=float(fm_params["sigma"]),
        covariate_specific_theta=False,
        plot_and_eval_every=10**9,
        use_ot=bool(fm_params["use_ot"]),
        is_binarized=False,
        modality_list=["rna"],
        guidance_weights={cov: 1},
    ).to(dev)

    # don’t force strict=True (CFGen versions differ); also ignore encoder weights here
    fm_sd2 = {k: v for k, v in fm_sd.items() if not k.startswith("encoder_model.")}
    keys = fm.load_state_dict(fm_sd2, strict=False)
    fm.eval()

    m = CFGenModel(device=device, workdir="__loaded__", covariate_key=cov)
    m.encoder_model = enc
    m.fm_model = fm
    m._meta = meta

    print("Loaded CFGen bundle (weights-only):", bundle_dir, "| zdim =", zdim,
          "| missing =", len(keys.missing_keys), "| unexpected =", len(keys.unexpected_keys))
    return m
