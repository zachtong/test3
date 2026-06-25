"""Error metrics for the 3D field."""

from __future__ import annotations
from typing import Iterable

import numpy as np

from core.simulation import Simulation
from core.pod_basis import PODBasis


def relative_l2_error(f_hat: np.ndarray, f_true: np.ndarray) -> float:
    """Shape-agnostic relative L2 norm; works for (Nx,Ny,Nt) and 1D vectors."""
    return float(np.linalg.norm(f_hat - f_true)
                 / (np.linalg.norm(f_true) + 1e-12))


def per_mode_rel_l2(pred: np.ndarray, true: np.ndarray,
                    axis: int = -1) -> np.ndarray:
    num = np.linalg.norm(pred - true, axis=axis)
    den = np.linalg.norm(true, axis=axis) + 1e-12
    return num / den


def truncation_floor(basis: PODBasis,
                     sims: Iterable[Simulation]) -> np.ndarray:
    """K-mode oracle floor: per-sim rel-L2 of (Phi Phi^T) f_true vs f_true.

    The lower bound the model can ever reach: it sees the true projection
    coefficients (not predicted ones), so this is purely "what fraction of
    the field does the K-mode subspace capture".
    """
    errs = []
    for s in sims:
        a = basis.project_sim(s)
        f_hat = basis.reconstruct(a)
        errs.append(relative_l2_error(f_hat, s.f))
    return np.array(errs)


def stats(arr: np.ndarray) -> dict:
    return {"n": int(len(arr)),
            "median": float(np.median(arr)),
            "p95": float(np.percentile(arr, 95)),
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr))}
