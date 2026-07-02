"""Trajectory-level cache: skip loading F entirely on subsequent trains.

The training pipeline only needs a small amount of data:
  - y (sensor time series): (n_sim, n_sensors, Nt) float32
  - a (POD projections):    (n_sim, K, Nt) float32
  - test_f_true_sq_norm, test_f_perp_sq_norm: per-test-sim scalars used
    to compute field_errs / floor_errs via Parseval WITHOUT ever
    touching F again.

At Nx=Ny=128, Nt=300, K=8, 5000 sims, that is ~66 MB on disk vs the
93 GB canonical F. Reading 66 MB is instant even on a spinning disk.

Cache is keyed by everything that changes y or a values:
  - npz_dir (resolved absolute path)
  - grid (nx, ny, nt, x_end, y_end)
  - drop_first_steps
  - split (seed, train_frac, val_frac)
  - loader rim mask (_DISK_MASK_R_END)  -- masks affect POD projections
  - sensor positions
  - K

Cache file: outputs/basis_cache/traj_<hash>.npz (co-located with the
POD basis cache -- both invalidate on the same axes and staying in
the same directory means one rm cleans both).
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

import numpy as np


def _traj_key(npz_dir, nx, ny, nt, x_end, y_end, drop_first_steps,
              seed, train_frac, val_frac, sensor_positions, K):
    """Stable short hash. `sensor_positions` is an iterable of (r, th)
    tuples in the config order (order matters -- swapping sensor 0
    and sensor 1 produces different y tensors, so a different key)."""
    from data.loader import _DISK_MASK_R_END
    try:
        npz_dir = str(Path(str(npz_dir)).expanduser().resolve())
    except (OSError, ValueError):
        npz_dir = str(npz_dir)
    sens_str = ",".join(f"({r:g},{th:g})" for r, th in sensor_positions)
    raw = (f"traj|{npz_dir}|{nx}|{ny}|{nt}|{x_end}|{y_end}|"
           f"{drop_first_steps}|{seed}|{train_frac}|{val_frac}|"
           f"rim={_DISK_MASK_R_END}|K={K}|sens=[{sens_str}]")
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def traj_cache_path(cache_dir, key: str) -> Path:
    return Path(cache_dir) / f"traj_{key}.npz"


def try_load_traj(path: Path):
    """Return a dict on hit, None on miss (including corruption)."""
    if not path.is_file():
        return None
    try:
        with np.load(path, allow_pickle=False) as z:
            return dict(
                x_canon=z["x_canon"].astype(np.float64),
                y_canon=z["y_canon"].astype(np.float64),
                sensor_xy=z["sensor_xy"].astype(np.float64),
                s_ij=z["s_ij"].astype(np.int64),
                y_train_val=z["y_train_val"].astype(np.float32),
                a_train_val=z["a_train_val"].astype(np.float32),
                y_test=z["y_test"].astype(np.float32),
                a_test=z["a_test"].astype(np.float32),
                test_f_true_sq_norm=z["test_f_true_sq_norm"]
                                     .astype(np.float64),
                test_f_perp_sq_norm=z["test_f_perp_sq_norm"]
                                     .astype(np.float64),
                test_basenames=json.loads(str(z["test_basenames_json"])),
                n_train=int(z["n_train"]),
                n_val=int(z["n_val"]))
    except (OSError, KeyError, ValueError) as e:
        print(f"  trajectory cache unreadable "
              f"({type(e).__name__}: {e}); rebuilding",
              flush=True)
        return None


def save_traj(path: Path, data: dict) -> None:
    """Atomic write. All fields float32 except the tiny scalar-per-sim
    norm arrays which stay float64 for scoring precision."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npz")
    np.savez(
        tmp,
        x_canon=np.asarray(data["x_canon"], dtype=np.float64),
        y_canon=np.asarray(data["y_canon"], dtype=np.float64),
        sensor_xy=np.asarray(data["sensor_xy"], dtype=np.float64),
        s_ij=np.asarray(data["s_ij"], dtype=np.int64),
        y_train_val=np.asarray(data["y_train_val"], dtype=np.float32),
        a_train_val=np.asarray(data["a_train_val"], dtype=np.float32),
        y_test=np.asarray(data["y_test"], dtype=np.float32),
        a_test=np.asarray(data["a_test"], dtype=np.float32),
        test_f_true_sq_norm=np.asarray(data["test_f_true_sq_norm"],
                                         dtype=np.float64),
        test_f_perp_sq_norm=np.asarray(data["test_f_perp_sq_norm"],
                                         dtype=np.float64),
        test_basenames_json=np.array(
            json.dumps(list(data["test_basenames"]))),
        n_train=np.int64(data["n_train"]),
        n_val=np.int64(data["n_val"]))
    tmp.replace(path)


def compute_test_field_norms(test_sims, a_test: np.ndarray,
                              progress_every: int = 500
                              ) -> tuple[np.ndarray, np.ndarray]:
    """Precompute the scalar norms scorer needs:

      ||f_true_j||^2   = full L2 norm of test sim j's canonical field
      ||f_perp_j||^2   = truncation error norm = ||f_true - Phi @ a_true||^2

    Using Parseval on an orthonormal Phi: ||f_true||^2 = ||a_true||^2
    + ||f_perp||^2, so f_perp_sq = ||f_true||^2 - ||a_test||^2.

    Both are (n_test,) float64.

    Perf note: `np.einsum('ijk,ijk->', s.f, s.f, dtype=np.float64)`
    accumulates in float64 without allocating a 39 MB float64 temp
    per call. Prior implementation did `np.asarray(s.f, float64)`
    which stalled the loop on memory bandwidth for large ensembles.
    """
    import time
    n = len(test_sims)
    f_true_sq = np.empty(n, dtype=np.float64)
    t0 = time.time()
    for i, s in enumerate(test_sims):
        f_true_sq[i] = float(np.einsum("ijk,ijk->",
                                         s.f, s.f, dtype=np.float64))
        if progress_every and ((i + 1) % progress_every == 0
                                 or i == n - 1):
            elapsed = time.time() - t0
            rate = (i + 1) / max(elapsed, 1e-9)
            eta = (n - i - 1) / max(rate, 1e-9)
            print(f"  f_norms: {i + 1}/{n}  "
                  f"({rate:.1f}/s  ETA {eta:.1f}s)", flush=True)
    a_norm_sq = np.einsum("ijk,ijk->i", a_test, a_test,
                           dtype=np.float64)
    f_perp_sq = np.maximum(f_true_sq - a_norm_sq, 0.0)
    return f_true_sq, f_perp_sq
