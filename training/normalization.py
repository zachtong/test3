"""Per-channel z-score normalization."""

from __future__ import annotations
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class NormStats:
    mean: np.ndarray
    std: np.ndarray


def compute_norm_stats(x: np.ndarray,
                       axes: tuple[int, ...] = (0, 2)) -> NormStats:
    return NormStats(mean=x.mean(axis=axes, keepdims=True),
                     std=x.std(axis=axes, keepdims=True) + 1e-6)


def apply_norm(x: np.ndarray, stats: NormStats) -> np.ndarray:
    return (x - stats.mean) / stats.std


def invert_norm(x: np.ndarray, stats: NormStats) -> np.ndarray:
    return x * stats.std + stats.mean


def save_norm_stats(path, y_stats: NormStats, target_stats: NormStats) -> None:
    """Persist the input/target normalization stats so inference can reuse them."""
    import numpy as _np
    _np.savez(path, y_mean=y_stats.mean, y_std=y_stats.std,
              target_mean=target_stats.mean, target_std=target_stats.std)


def load_norm_stats(path) -> tuple[NormStats, NormStats]:
    with np.load(path) as z:
        return (NormStats(z["y_mean"], z["y_std"]),
                NormStats(z["target_mean"], z["target_std"]))
