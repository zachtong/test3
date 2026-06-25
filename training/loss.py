"""Loss functions and channel weight construction."""

from __future__ import annotations

import numpy as np
import torch


def make_channel_weights(sigma: np.ndarray, scheme: str,
                         with_front: bool = True) -> torch.Tensor:
    """Build the per-channel weight vector for the target.

    With a front channel (sPOD) the result is (1+K,) with w[0]=1 for rb; without
    one (POD) it is (K,) over the coefficients only.
    """
    if scheme == "sigma2":
        w_a = (sigma ** 2) / (sigma ** 2).sum()
    elif scheme == "sigma":
        w_a = sigma / sigma.sum()
    elif scheme == "sqrt_sigma":
        sq = np.sqrt(sigma)
        w_a = sq / sq.sum()
    elif scheme == "uniform":
        w_a = np.ones_like(sigma) / len(sigma)
    else:
        raise ValueError(f"unknown weight scheme: {scheme!r}")
    w = np.concatenate(([1.0], w_a)) if with_front else w_a
    return torch.tensor(w, dtype=torch.float32)


def channel_weighted_mse(pred: torch.Tensor, target: torch.Tensor,
                         weights: torch.Tensor) -> torch.Tensor:
    per_channel = (pred - target).pow(2).mean(dim=(0, 2))
    return (per_channel * weights).sum()
