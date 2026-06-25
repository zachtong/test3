"""Dataset construction: sims + sensor extraction + POD projection.

3D / POD-only. The pipeline:
  - place_sensors(cfg) -> (n, 2) Cartesian xy from the polar config
  - sensor_indices(xy, x_canon, y_canon) -> (n, 2) integer (ix, iy)
  - extract_batch(sims, ij) -> (B, n, Nt) sensor traces (y)
  - basis.project_ensemble(sims) -> (B, K, Nt) POD coefficients (target)
"""

from __future__ import annotations

import numpy as np
import torch

from core.simulation import Simulation
from core.pod_basis import PODBasis
from core.sensors import SensorConfig, place_sensors, sensor_indices, extract_batch
from training.normalization import NormStats, compute_norm_stats, apply_norm


def build_trajectory_dataset(
    sims: list[Simulation], x_canon: np.ndarray, y_canon: np.ndarray,
    basis: PODBasis, sensor_cfg: SensorConfig,
) -> dict:
    sensor_xy = place_sensors(sensor_cfg)
    s_ij = sensor_indices(sensor_xy, x_canon, y_canon)
    y = extract_batch(sims, s_ij)
    target = basis.project_ensemble(sims)
    return dict(sensor_xy=sensor_xy, s_ij=s_ij, y=y,
                a=target, target=target)


def normalize_dataset(ds: dict, y_stats: NormStats | None = None,
                      target_stats: NormStats | None = None,
                      device: torch.device | None = None) -> dict:
    if y_stats is None:
        y_stats = compute_norm_stats(ds["y"])
    if target_stats is None:
        target_stats = compute_norm_stats(ds["target"])
    dev = device or torch.device("cpu")
    return dict(
        sensor_xy=ds["sensor_xy"], s_ij=ds["s_ij"], y_raw=ds["y"],
        y_t=torch.tensor(apply_norm(ds["y"], y_stats),
                         dtype=torch.float32, device=dev),
        target_t=torch.tensor(apply_norm(ds["target"], target_stats),
                              dtype=torch.float32, device=dev),
        y_stats=y_stats, target_stats=target_stats,
    )
