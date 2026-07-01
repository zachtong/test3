"""Reconstruct predicted / ground-truth / oracle-POD fields for a trained 3D run.

3D analogue of the 2D evaluation/run_predict.py. The run's config --
sensors, K, grid, drop_first_steps -- is recovered from
outputs/<tag>/results.json so it always matches the trained model and
cannot silently drift; only the data path / output dirs may be
overridden (e.g. on a different machine).

POD-only: no rb / front channel; basis.reconstruct takes just `a` of
shape (K, Nt).
"""

from __future__ import annotations
from pathlib import Path

import numpy as np
import torch

from training.config import config_with_overrides, _build_sub, ExperimentConfig
from training.dataset import build_trajectory_dataset, normalize_dataset
from training.checkpoint import (checkpoint_path, history_path,
                                  load_checkpoint)
from training.normalization import invert_norm
from training.basis_cache import load_or_fit_basis
from core.pod_basis import PODBasis
from core.sensors import SensorConfig, place_sensors
from data.loader import load_dataset
from data.splitter import split_dataset
from models import create_model
from evaluation.result import ResultSet


def load_run_config(tag, *, output_dir="outputs", overrides=None):
    """Rebuild the exact ExperimentConfig a run was trained with."""
    rs_path = Path(output_dir) / tag / "results.json"
    if not rs_path.exists():
        raise FileNotFoundError(
            f"no results.json for tag {tag!r} at {rs_path}")
    cfg = _build_sub(ExperimentConfig,
                     ResultSet.load_json(str(rs_path)).config)
    keep = {"output_dir": output_dir}
    keep.update({k: v for k, v in (overrides or {}).items()
                 if k in ("data.npz_dir", "data.limit",
                          "basis_cache_dir")})
    return config_with_overrides(cfg, keep)


def _load_basis_file(path, K: int, verbose: bool = True) -> PODBasis:
    """Direct-load a POD basis file, bypassing the cache-key lookup.

    Use when the caller KNOWS a specific pod3d_*.npz is the right
    basis for this run (e.g. it was fit during training) but the
    basis_cache key derivation would produce a MISS due to a subtle
    config-field drift between training and inference.

    Slices the file down to the requested K modes; raises if the file
    has fewer than K.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"--basis-file not found: {p}")
    with np.load(p, allow_pickle=False) as z:
        k_stored = int(z["k_cache"])
        if k_stored < K:
            raise ValueError(
                f"--basis-file {p.name} has k_cache={k_stored} < "
                f"requested K={K}; cannot slice up. Re-fit at "
                f"k_cache >= {K} or use a different file.")
        Phi = z["Phi"][:, :K].copy()
        sigma = z["sigma"][:K].copy()
        spatial_shape = tuple(int(d) for d in z["spatial_shape"])
    if verbose:
        print(f"  POD basis LOADED (override) <- {p.name} "
              f"(k_stored={k_stored}, sliced to K={K})", flush=True)
    return PODBasis(Phi=Phi, sigma=sigma, spatial_shape=spatial_shape)


def predict_run_fields(tag, *, idx=None, n_samples=5, sample_seed=0,
                       output_dir="outputs", overrides=None, device=None,
                       verbose=True, basis_override_path=None):
    """Predict + reconstruct fields for some test sims of a trained run.

    idx: explicit test indices into the run's test split (same sim
    across runs that share the same data + seed). If None, picks
    n_samples random ones.

    Returns a dict with: x_canon, y_canon, t, sensor_xy, K, idx,
    w_pred (Nx, Ny, Nt) per sim, w_true, w_pod (oracle K-mode), cfg,
    test_sims_meta (basenames for the picked sims).
    """
    cfg = load_run_config(tag, output_dir=output_dir, overrides=overrides)
    device = device or torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    if verbose:
        print(f"[{tag}] K={cfg.pod.k} sensors n={cfg.sensors.n} "
              f"positions={list(cfg.sensors.positions)} "
              f"drop_first_steps={cfg.data.drop_first_steps}")

    x_canon, y_canon, sims = load_dataset(
        cfg.data.npz_dir, nx=cfg.data.nx, ny=cfg.data.ny, nt=cfg.data.nt,
        x_end=cfg.data.x_end, y_end=cfg.data.y_end,
        limit=cfg.data.limit, workers=cfg.data.workers,
        drop_first_steps=cfg.data.drop_first_steps)
    split = split_dataset(sims, train_frac=cfg.data.train_frac,
                          val_frac=cfg.train.val_frac, seed=cfg.data.seed)
    train_sims, val_sims = split["train_sims"], split["val_sims"]
    test_sims = split["test_sims"]
    if verbose:
        print(f"  split: {len(train_sims)}/{len(val_sims)}/{len(test_sims)}")

    if basis_override_path is not None:
        basis = _load_basis_file(basis_override_path, K=cfg.pod.k,
                                   verbose=verbose)
    else:
        basis = load_or_fit_basis(
            train_sims + val_sims, K=cfg.pod.k,
            npz_dir=cfg.data.npz_dir, nx=cfg.data.nx, ny=cfg.data.ny,
            nt=cfg.data.nt, x_end=cfg.data.x_end, y_end=cfg.data.y_end,
            drop_first_steps=cfg.data.drop_first_steps,
            seed=cfg.data.seed, train_frac=cfg.data.train_frac,
            val_frac=cfg.train.val_frac,
            cache_dir=cfg.basis_cache_dir, k_cache=cfg.pod.k_cache,
            workers=cfg.pod.workers)

    scfg = SensorConfig(n=cfg.sensors.n, strategy=cfg.sensors.strategy,
                        positions=cfg.sensors.positions)
    sensor_xy = place_sensors(scfg)
    ds_tr = normalize_dataset(
        build_trajectory_dataset(train_sims + val_sims, x_canon, y_canon,
                                  basis, scfg),
        device=device)
    ds_te = normalize_dataset(
        build_trajectory_dataset(test_sims, x_canon, y_canon, basis, scfg),
        y_stats=ds_tr["y_stats"], target_stats=ds_tr["target_stats"],
        device=device)

    preds = []
    for seed in cfg.seeds:
        cp = checkpoint_path(cfg.output_dir, tag, seed)
        if not cp.exists():
            continue
        model = create_model(
            cfg.model.arch, n_in=cfg.sensors.n, n_out=cfg.pod.k,
            channels=cfg.model.channels, dilations=cfg.model.dilations,
            kernel=cfg.model.kernel, dropout=cfg.model.dropout,
            causal=cfg.model.causal).to(device)
        load_checkpoint(model, cp,
                        history_path(cfg.output_dir, tag, seed), device)
        model.eval()
        with torch.no_grad():
            preds.append(model(ds_te["y_t"]).cpu().numpy())
    if not preds:
        raise SystemExit(f"no checkpoints found for tag {tag!r}")
    Y = invert_norm(np.mean(np.stack(preds), axis=0),
                    ds_tr["target_stats"])

    if idx is None:
        rng = np.random.default_rng(sample_seed)
        idx = rng.choice(len(test_sims),
                         size=min(n_samples, len(test_sims)),
                         replace=False)
    idx = np.asarray(idx)
    w_pred, w_true, w_pod = [], [], []
    metas = []
    for i in idx:
        a_pred = Y[i]                                      # (K, Nt)
        w_pred.append(basis.reconstruct(a_pred))
        a_true = basis.project_sim(test_sims[i])
        w_pod.append(basis.reconstruct(a_true))
        w_true.append(np.asarray(test_sims[i].f, dtype=np.float64))
        metas.append(test_sims[i].params.get("basename", f"sim_{int(i)}"))

    return dict(
        x_canon=x_canon, y_canon=y_canon,
        t=np.linspace(0.0, 1.0, cfg.data.nt),
        sensor_xy=sensor_xy, K=cfg.pod.k, idx=idx,
        w_pred=np.stack(w_pred), w_true=np.stack(w_true),
        w_pod=np.stack(w_pod),
        cfg=cfg, basenames=metas,
        a_pred=np.stack([Y[i] for i in idx]),
        a_true=np.stack([basis.project_sim(test_sims[i]) for i in idx]),
    )
