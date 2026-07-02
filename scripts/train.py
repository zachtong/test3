"""Train POD + BiTCN on the 3D Cartesian field.

    python scripts/train.py --config configs/default.yaml --data.npz_dir /path
    python scripts/train.py --config configs/default.yaml --model.channels 128

CLI overrides follow the same dotted form as the 2D codebase (e.g.
`--pod.k 4 --sensors.n 8`). Values are JSON-decoded so e.g.
`--sensors.positions "[[1.0, 0], [1.0, 90]]"` works.
"""

from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import numpy as np
import torch

from training.config import config_from_yaml, config_with_overrides
from training.dataset import build_trajectory_dataset, normalize_dataset
from training.loss import make_channel_weights
from training.loop import train_one_seed
from training.checkpoint import (checkpoint_path, history_path, latest_path,
                                  save_checkpoint, load_checkpoint)
from training.normalization import save_norm_stats
from training.basis_cache import load_or_fit_basis, load_cached_basis
from training.traj_cache import (_traj_key, traj_cache_path,
                                   try_load_traj, save_traj,
                                   compute_test_field_norms)
from core.sensors import SensorConfig
from data.loader import load_dataset
from data.splitter import split_dataset
from models import create_model
from evaluation.scorer import (evaluate_ensemble,
                                 evaluate_from_cached_norms)


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args, unknown = parser.parse_known_args()

    overrides: dict = {}
    i = 0
    while i < len(unknown):
        if unknown[i].startswith("--"):
            key = unknown[i][2:]
            val = unknown[i + 1] if i + 1 < len(unknown) else ""
            try:
                val = json.loads(val)
            except (json.JSONDecodeError, TypeError):
                pass
            overrides[key] = val
            i += 2
        else:
            i += 1

    cfg = config_from_yaml(args.config)
    if overrides:
        cfg = config_with_overrides(cfg, overrides)

    device = pick_device()
    tag = cfg.tag or "default"
    out_dir = Path(cfg.output_dir) / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"device={device}  tag={tag}")
    print(f"POD K={cfg.pod.k}  n={cfg.sensors.n}  {cfg.model.arch} "
          f"{cfg.model.channels}ch  {cfg.train.epochs}ep  seeds={cfg.seeds}")

    if not cfg.data.npz_dir:
        raise SystemExit(
            "data.npz_dir is required. The 3D pipeline has no synthetic "
            "fallback yet; point at a converted-NPZ folder.")

    # Trajectory-level cache. Everything training + scoring needs
    # (sensor traces, POD projections, and the two scalar-per-test-sim
    # norms scorer uses via Parseval) fits in ~66 MB. A HIT skips the
    # 93 GB F reload from load_dataset entirely -- next-day retrains
    # after a rebooted server take seconds to reach the model loop
    # instead of dozens of minutes. Cache key covers every param that
    # changes y or a (npz_dir + grid + drop + split + rim mask +
    # sensor positions + K); ANY of those changing forces a rebuild.
    scfg = SensorConfig(n=cfg.sensors.n, strategy=cfg.sensors.strategy,
                        positions=cfg.sensors.positions)
    traj_k = _traj_key(
        cfg.data.npz_dir, cfg.data.nx, cfg.data.ny, cfg.data.nt,
        cfg.data.x_end, cfg.data.y_end, cfg.data.drop_first_steps,
        cfg.data.seed, cfg.data.train_frac, cfg.train.val_frac,
        cfg.sensors.positions, cfg.pod.k)
    traj_path = traj_cache_path(cfg.basis_cache_dir, traj_k)
    traj = None if cfg.pod.force_refit else try_load_traj(traj_path)

    if traj is not None:
        print(f"trajectory cache HIT -> {traj_path.name} "
              f"(skipping load_dataset + build_trajectory_dataset)")
        x_canon = traj["x_canon"]; y_canon = traj["y_canon"]
        n_train = traj["n_train"]
        # Build the same dict shape build_trajectory_dataset returns
        # so downstream code (normalize_dataset) is unchanged.
        ds_tr_raw = dict(sensor_xy=traj["sensor_xy"], s_ij=traj["s_ij"],
                          y=traj["y_train_val"],
                          a=traj["a_train_val"],
                          target=traj["a_train_val"])
        ds_te_raw = dict(sensor_xy=traj["sensor_xy"], s_ij=traj["s_ij"],
                          y=traj["y_test"], a=traj["a_test"],
                          target=traj["a_test"])
        test_basenames = traj["test_basenames"]
        test_f_true_sq_norm = traj["test_f_true_sq_norm"]
        test_f_perp_sq_norm = traj["test_f_perp_sq_norm"]
        # Basis still needed for weight_scheme sigma calculation and
        # any downstream POD reconstruction. Load directly from the
        # basis cache using n_fit from the traj cache (does not
        # touch sims).
        basis = load_cached_basis(
            npz_dir=cfg.data.npz_dir, nx=cfg.data.nx, ny=cfg.data.ny,
            nt=cfg.data.nt, x_end=cfg.data.x_end, y_end=cfg.data.y_end,
            drop_first_steps=cfg.data.drop_first_steps,
            seed=cfg.data.seed, train_frac=cfg.data.train_frac,
            val_frac=cfg.train.val_frac,
            n_fit=n_train + traj["n_val"],
            cache_dir=cfg.basis_cache_dir, K=cfg.pod.k)
        if basis is None:
            raise SystemExit(
                "trajectory cache HIT but its companion basis file is "
                "missing from basis_cache. Delete traj_*.npz to force "
                "a full rebuild.")
        test_sims = None    # scoring will use cached norms
        used_cache = True
    else:
        # data
        t0 = time.time()
        x_canon, y_canon, sims = load_dataset(
            cfg.data.npz_dir, nx=cfg.data.nx, ny=cfg.data.ny,
            nt=cfg.data.nt, x_end=cfg.data.x_end, y_end=cfg.data.y_end,
            limit=cfg.data.limit, workers=cfg.data.workers,
            drop_first_steps=cfg.data.drop_first_steps)
        print(f"data: {len(sims)} sims ({time.time() - t0:.1f}s)")

        split = split_dataset(
            sims, train_frac=cfg.data.train_frac,
            val_frac=cfg.train.val_frac, seed=cfg.data.seed)
        train_sims = split["train_sims"]
        val_sims = split["val_sims"]
        test_sims = split["test_sims"]
        n_train = len(train_sims)
        print(f"split: {n_train}/{len(val_sims)}/{len(test_sims)}")

        basis = load_or_fit_basis(
            train_sims + val_sims, K=cfg.pod.k,
            npz_dir=cfg.data.npz_dir, nx=cfg.data.nx, ny=cfg.data.ny,
            nt=cfg.data.nt, x_end=cfg.data.x_end, y_end=cfg.data.y_end,
            drop_first_steps=cfg.data.drop_first_steps,
            seed=cfg.data.seed, train_frac=cfg.data.train_frac,
            val_frac=cfg.train.val_frac,
            cache_dir=cfg.basis_cache_dir, k_cache=cfg.pod.k_cache,
            force_refit=cfg.pod.force_refit,
            workers=cfg.pod.workers)

        ds_tr_raw = build_trajectory_dataset(
            train_sims + val_sims, x_canon, y_canon, basis, scfg)
        ds_te_raw = build_trajectory_dataset(
            test_sims, x_canon, y_canon, basis, scfg)
        test_basenames = [s.params.get("basename", "")
                           for s in test_sims]
        # Compute the scalar-per-test-sim norms scorer needs so we
        # can score without re-loading F next time.
        test_f_true_sq_norm, test_f_perp_sq_norm = (
            compute_test_field_norms(test_sims, ds_te_raw["a"]))
        try:
            save_traj(traj_path, dict(
                x_canon=x_canon, y_canon=y_canon,
                sensor_xy=ds_tr_raw["sensor_xy"],
                s_ij=ds_tr_raw["s_ij"],
                y_train_val=ds_tr_raw["y"],
                a_train_val=ds_tr_raw["a"],
                y_test=ds_te_raw["y"], a_test=ds_te_raw["a"],
                test_f_true_sq_norm=test_f_true_sq_norm,
                test_f_perp_sq_norm=test_f_perp_sq_norm,
                test_basenames=test_basenames,
                n_train=n_train, n_val=len(val_sims)))
            print(f"  wrote trajectory cache -> "
                  f"{traj_path.name} "
                  f"(~{(traj_path.stat().st_size >> 20)} MB)")
        except OSError as e:
            print(f"  WARN: could not write trajectory cache "
                  f"({type(e).__name__}: {e}); continuing")
        used_cache = False

    print(f"POD K={cfg.pod.k}  sigma ratio="
          f"{(basis.sigma / basis.sigma[0]).round(4).tolist()}")

    ds_tr = normalize_dataset(ds_tr_raw, device=device)
    ds_te = normalize_dataset(
        ds_te_raw,
        y_stats=ds_tr["y_stats"], target_stats=ds_tr["target_stats"],
        device=device)
    save_norm_stats(out_dir / "norm_stats.npz", ds_tr["y_stats"],
                    ds_tr["target_stats"])

    x_tr, y_tr = ds_tr["y_t"][:n_train], ds_tr["target_t"][:n_train]
    x_vl, y_vl = ds_tr["y_t"][n_train:], ds_tr["target_t"][n_train:]

    weights = make_channel_weights(basis.sigma, cfg.train.weight_scheme,
                                   with_front=False).to(device)

    # train
    models = []
    for seed in cfg.seeds:
        cp = checkpoint_path(cfg.output_dir, tag, seed)
        hp = history_path(cfg.output_dir, tag, seed)
        lp = latest_path(cfg.output_dir, tag, seed)
        model = create_model(
            cfg.model.arch, n_in=cfg.sensors.n, n_out=cfg.pod.k,
            channels=cfg.model.channels, dilations=cfg.model.dilations,
            kernel=cfg.model.kernel, dropout=cfg.model.dropout,
            causal=cfg.model.causal).to(device)
        if cp.exists():
            print(f"seed={seed}: cached")
            load_checkpoint(model, cp, hp, device)
        else:
            print(f"seed={seed}: training ...")
            model, _ = train_one_seed(
                model, x_tr, y_tr, x_vl, y_vl,
                weights, cfg.train, device, seed,
                save_fn=save_checkpoint, best_path=cp, latest_path=lp,
                hist_path=hp)
            save_checkpoint(model, _, cp, hp)
        models.append(model)

    # eval
    if used_cache:
        # Score via cached f-norms + a_true; no F needed.
        a_test_true_np = ds_te_raw["a"].astype(np.float64)
        result = evaluate_from_cached_norms(
            models, ds_te["y_t"], a_test_true_np,
            test_f_true_sq_norm, test_f_perp_sq_norm,
            test_basenames, basis.sigma,
            ds_tr["target_stats"], cfg)
    else:
        result = evaluate_ensemble(
            models, ds_te["y_t"], test_sims, basis,
            ds_tr["target_stats"], cfg)
    result.save_json(out_dir / "results.json")

    g = result.global_stats
    print(f"\nmed={g['median']:.4f}  p95={g['p95']:.4f}  "
          f"gap={result.gap_to_floor:.2f}x  "
          f"params={result.n_params:,}")
    print(f"saved: {out_dir / 'results.json'}")


if __name__ == "__main__":
    main()
