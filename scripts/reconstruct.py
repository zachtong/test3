"""Standalone inference: reconstruct the full quarter-disk field from sensors.

Loads a bundle (scripts/bundle.py) and reconstructs w(x, y, t) from sparse
sensor time series -- no training data, no caches, only the bundle + this
repo's model/basis code. POD-only (no sPOD / co-moving front). CPU,
milliseconds.

As a library:
    from scripts.reconstruct import load_bundle, reconstruct_field
    b = load_bundle("bundles/pod_k12_n6.pt")
    w = reconstruct_field(b, y, t_raw=t)   # y: (n, T) physical sensor readings
                                           # -> w: (Nx, Ny, Nt) quarter-disk

As a CLI (sensors npz must hold 'y' (n,T) and optionally 't' (T,)):
    python scripts/reconstruct.py --bundle bundles/pod_k12_n6.pt \
        --sensors my_run.npz --out field.npz
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

from models import create_model
from core.pod_basis import PODBasis
from training.normalization import NormStats, apply_norm, invert_norm


def load_bundle(path):
    # the bundle holds numpy arrays + a dict, so weights_only must be False
    return torch.load(str(path), map_location="cpu", weights_only=False)


def _build_models(b):
    m = b["model"]
    models = []
    for sd in b["state_dicts"]:
        mod = create_model(
            m["arch"], n_in=m["n_in"], n_out=m["n_out"],
            channels=m["channels"], dilations=tuple(m["dilations"]),
            kernel=m["kernel"], dropout=m["dropout"], causal=m["causal"])
        mod.load_state_dict(sd)
        mod.eval()
        models.append(mod)
    return models


def _resample(y, t_raw, nt):
    """Resample each sensor series onto the nt-point normalized-time grid.

    Matches the loader's per-run time normalization s = (t - t0)/(t_end - t0),
    so a sim-trained model transfers to arbitrarily-sampled real data.
    """
    t_raw = np.asarray(t_raw, dtype=float)
    span = t_raw[-1] - t_raw[0]
    s = (t_raw - t_raw[0]) / (span if span > 0 else 1.0)
    grid = np.linspace(0.0, 1.0, nt)
    return np.stack([np.interp(grid, s, y[i]) for i in range(y.shape[0])])


def reconstruct_field(bundle, y, t_raw=None):
    """y: (n, T) PHYSICAL sensor readings. t_raw: (T,) real times, or None if y
    is already on the nt-point normalized grid. Returns w_pred (Nx, Ny, Nt)."""
    b = bundle
    nt = int(b["nt"])
    y = np.asarray(y, dtype=float)
    if t_raw is not None and y.shape[1] != nt:
        y = _resample(y, t_raw, nt)
    yn = apply_norm(y[None], NormStats(b["y_mean"], b["y_std"]))    # (1, n, nt)
    x = torch.tensor(yn, dtype=torch.float32)
    with torch.no_grad():
        preds = [m(x).cpu().numpy() for m in _build_models(b)]
    Y = invert_norm(np.mean(np.stack(preds), axis=0),
                    NormStats(b["target_mean"], b["target_std"]))[0]  # (K, nt)
    spatial_shape = tuple(int(s) for s in b["spatial_shape"])
    basis = PODBasis(np.asarray(b["Phi"]), np.asarray(b["sigma"]),
                     spatial_shape)
    return basis.reconstruct(Y)                              # (Nx, Ny, Nt)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--sensors", required=True,
                    help="npz with 'y' (n,T) and optionally 't' (T,)")
    ap.add_argument("--out", default="field.npz")
    a = ap.parse_args()
    b = load_bundle(a.bundle)
    with np.load(a.sensors) as z:
        y = z["y"]
        t_raw = z["t"] if "t" in z.files else None
    w = reconstruct_field(b, y, t_raw=t_raw)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, w=w, x=np.asarray(b["x_canon"]), y=np.asarray(b["y_canon"]),
             t=np.linspace(0.0, 1.0, int(b["nt"])))
    print(f"reconstructed field {w.shape} -> {out}")


if __name__ == "__main__":
    main()
