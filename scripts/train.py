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
from core.sensors import SensorConfig
from core.pod_basis import PODBasis
from data.loader import load_dataset
from data.splitter import split_dataset
from models import create_model
from evaluation.scorer import evaluate_ensemble


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

    # data
    t0 = time.time()
    x_canon, y_canon, sims = load_dataset(
        cfg.data.npz_dir, nx=cfg.data.nx, ny=cfg.data.ny, nt=cfg.data.nt,
        x_end=cfg.data.x_end, y_end=cfg.data.y_end, limit=cfg.data.limit,
        workers=cfg.data.workers)
    print(f"data: {len(sims)} sims ({time.time() - t0:.1f}s)")

    split = split_dataset(sims, train_frac=cfg.data.train_frac,
                          val_frac=cfg.train.val_frac, seed=cfg.data.seed)
    train_sims = split["train_sims"]
    val_sims = split["val_sims"]
    test_sims = split["test_sims"]
    print(f"split: {len(train_sims)}/{len(val_sims)}/{len(test_sims)}")

    # basis: lab-frame POD, fit on train + val
    t0 = time.time()
    basis = PODBasis.fit(train_sims + val_sims, K=cfg.pod.k)
    print(f"POD K={cfg.pod.k}  sigma ratio="
          f"{(basis.sigma / basis.sigma[0]).round(4).tolist()}  "
          f"({time.time() - t0:.1f}s)")

    # datasets (sensor_cfg mirrors training.config.SensorConfig)
    scfg = SensorConfig(n=cfg.sensors.n, strategy=cfg.sensors.strategy,
                        positions=cfg.sensors.positions)
    ds_tr = normalize_dataset(
        build_trajectory_dataset(train_sims + val_sims, x_canon, y_canon,
                                  basis, scfg), device=device)
    ds_te = normalize_dataset(
        build_trajectory_dataset(test_sims, x_canon, y_canon, basis, scfg),
        y_stats=ds_tr["y_stats"], target_stats=ds_tr["target_stats"],
        device=device)
    save_norm_stats(out_dir / "norm_stats.npz", ds_tr["y_stats"],
                    ds_tr["target_stats"])

    n_tr = len(train_sims)
    x_tr, y_tr = ds_tr["y_t"][:n_tr], ds_tr["target_t"][:n_tr]
    x_vl, y_vl = ds_tr["y_t"][n_tr:], ds_tr["target_t"][n_tr:]

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
