"""Package a trained 3D run into a self-contained inference bundle (one .pt).

POD-only (no sPOD / co-moving front). DATASET-FREE: the basis is read straight
from a pod3d_*.npz basis file (--basis-file), not by re-loading the training
data, so this runs anywhere the run's outputs/<tag>/ dir and the basis file are
present (e.g. a laptop). The run's config -- sensors, K, grid, model spec -- is
recovered from outputs/<tag>/results.json so it always matches training and
cannot drift.

The bundle holds everything scripts/reconstruct.reconstruct_field needs: the
POD basis (Phi, sigma, spatial grid), all-seed model weights, the input/target
normalization stats, the sensor (r, theta) + (x, y) + (ix, iy) positions, and
grid/model metadata. NO training data or caches at inference.

    python scripts/bundle.py --tag merged_sweep_k12_n6_ABCDEF \
        --basis-file outputs/basis_cache/pod3d_<key>.npz \
        --out bundles/pod_k12_n6.pt
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import numpy as np
import torch

from evaluation.run_predict import load_run_config
from core.grid import canonical_grid
from core.sensors import SensorConfig, place_sensors, sensor_indices
from training.normalization import load_norm_stats
from training.checkpoint import checkpoint_path


def _load_basis_file(path, K: int):
    """Direct-load a pod3d_*.npz basis, sliced to the leading K modes.

    POD modes are nested in K, so a cache fit at k_cache >= K serves any K.
    Returns (Phi (Nx*Ny, K), sigma (K,), spatial_shape (Nx, Ny))."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"--basis-file not found: {p}")
    with np.load(p, allow_pickle=False) as z:
        k_stored = int(z["k_cache"])
        if k_stored < K:
            raise ValueError(
                f"basis file {p.name} has k_cache={k_stored} < requested "
                f"K={K}; re-fit at k_cache >= {K} or use another file.")
        Phi = z["Phi"][:, :K].copy()
        sigma = z["sigma"][:K].copy()
        spatial_shape = tuple(int(d) for d in z["spatial_shape"])
    return Phi, sigma, spatial_shape


def build_bundle(tag: str, basis_file: str, out: str,
                 output_dir: str = "outputs") -> dict:
    """Assemble and save the self-contained inference bundle; return it."""
    cfg = load_run_config(tag, output_dir=output_dir)
    K = int(cfg.pod.k)
    nx, ny, nt = int(cfg.data.nx), int(cfg.data.ny), int(cfg.data.nt)
    x_canon, y_canon = canonical_grid(nx, ny, cfg.data.x_end, cfg.data.y_end)

    Phi, sigma, spatial_shape = _load_basis_file(basis_file, K)
    if spatial_shape != (nx, ny):
        raise ValueError(
            f"basis spatial_shape {spatial_shape} != config grid "
            f"({nx}, {ny}); wrong basis file for this run?")

    scfg = SensorConfig(n=cfg.sensors.n, strategy=cfg.sensors.strategy,
                        positions=cfg.sensors.positions)
    sensor_xy = place_sensors(scfg)                       # (n, 2)
    sensor_ij = sensor_indices(sensor_xy, x_canon, y_canon)   # (n, 2)
    rtheta = np.asarray(cfg.sensors.positions, dtype=np.float64).reshape(-1, 2)

    norm_path = Path(output_dir) / tag / "norm_stats.npz"
    if not norm_path.is_file():
        raise SystemExit(
            f"missing {norm_path}: the 3D trainer saves norm_stats.npz every "
            f"run; the run is incomplete or the tag is wrong.")
    y_stats, target_stats = load_norm_stats(norm_path)

    state_dicts = []
    for seed in cfg.seeds:
        cp = checkpoint_path(cfg.output_dir, tag, seed)
        if cp.is_file():
            state_dicts.append(torch.load(cp, map_location="cpu"))
    if not state_dicts:
        raise SystemExit(f"no checkpoints for tag {tag!r} under {output_dir}/")

    bundle = {
        "tag": tag,
        "uses_front": False,
        "Phi": np.asarray(Phi), "sigma": np.asarray(sigma),
        "spatial_shape": np.asarray(spatial_shape, dtype=np.int64),
        "x_canon": np.asarray(x_canon), "y_canon": np.asarray(y_canon),
        "nx": nx, "ny": ny, "nt": nt, "K": K,
        "x_end": float(cfg.data.x_end), "y_end": float(cfg.data.y_end),
        "sensor_rtheta": rtheta,
        "sensor_xy": np.asarray(sensor_xy),
        "sensor_ij": np.asarray(sensor_ij),
        "y_mean": y_stats.mean, "y_std": y_stats.std,
        "target_mean": target_stats.mean, "target_std": target_stats.std,
        "model": {"arch": cfg.model.arch, "n_in": int(cfg.sensors.n),
                  "n_out": K, "channels": int(cfg.model.channels),
                  "dilations": list(cfg.model.dilations),
                  "kernel": int(cfg.model.kernel),
                  "dropout": float(cfg.model.dropout),
                  "causal": bool(cfg.model.causal)},
        "seeds": list(cfg.seeds), "state_dicts": state_dicts,
    }
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, out_path)
    sz = out_path.stat().st_size / 1e6
    print(f"saved bundle {out_path} ({sz:.1f} MB): K={K} n={cfg.sensors.n} "
          f"grid=({nx},{ny}) seeds={len(state_dicts)} "
          f"sensor_rtheta={np.round(rtheta, 3).tolist()}")
    return bundle


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tag", required=True)
    ap.add_argument("--basis-file", required=True,
                    help="the pod3d_*.npz basis fit for this run")
    ap.add_argument("--out", required=True)
    ap.add_argument("--output-dir", default="outputs",
                    help="dir with <tag>/results.json, norm_stats.npz, "
                    "checkpoints/ (default: outputs)")
    a = ap.parse_args()
    build_bundle(a.tag, a.basis_file, a.out, output_dir=a.output_dir)


if __name__ == "__main__":
    main()
