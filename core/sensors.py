"""Sensor placement and signal extraction on the 3D Cartesian grid.

Sensor positions enter the config as polar (r, theta_deg) pairs because the
rig itself places sensors at radii on named axes (X, D=45deg, ...). At use
time each (r, theta) is converted to a Cartesian point, then snapped to the
nearest grid cell (ix, iy). The field is sampled at those cells; n traces of
length Nt feed the model.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from core.simulation import Simulation
from core.grid import canonical_grid, polar_to_xy, xy_to_indices


@dataclass(frozen=True)
class SensorConfig:
    """Runtime sensor config -- mirror of training.config.SensorConfig.

    Two copies exist on purpose: the training one is a frozen-dataclass YAML
    target (pure data), this one is the runtime form passed into placement /
    indexing helpers. They are kept in sync by `scripts/train.py`.
    """
    n: int = 4
    strategy: str = "custom"
    positions: tuple[tuple[float, float], ...] = ()


def place_sensors(cfg: SensorConfig) -> np.ndarray:
    """Return (n, 2) array of Cartesian (x, y) sensor positions.

    Only the "custom" strategy is supported today; positions must be a tuple
    of (r, theta_deg) pairs of length `cfg.n`. Ring / log strategies can be
    added later but are deferred until the rig schedule is known.
    """
    if cfg.strategy != "custom":
        raise ValueError(f"unsupported strategy {cfg.strategy!r}; "
                         f"only 'custom' is implemented")
    if len(cfg.positions) != cfg.n:
        raise ValueError(f"need {cfg.n} (r, theta) positions, "
                         f"got {len(cfg.positions)}")
    xy = np.array([polar_to_xy(r, th) for r, th in cfg.positions],
                  dtype=np.float64)
    return xy


def sensor_indices(sensor_xy: np.ndarray, x_grid: np.ndarray,
                   y_grid: np.ndarray) -> np.ndarray:
    """Map (n, 2) sensor xy to (n, 2) integer (ix, iy) indices."""
    return np.array([xy_to_indices(x, y, x_grid, y_grid)
                     for x, y in sensor_xy], dtype=np.int64)


def extract_batch(sims: Iterable[Simulation],
                  sensor_ij: np.ndarray) -> np.ndarray:
    """Pull (B, n, Nt) sensor traces from sims given their (n, 2) (ix, iy).

    Each simulation's `f` is (Nx, Ny, Nt); fancy-indexing the spatial axes
    with the sensor (ix, iy) pairs yields (n, Nt) traces per sim.
    """
    ix, iy = sensor_ij[:, 0], sensor_ij[:, 1]
    return np.stack([s.f[ix, iy, :] for s in sims], axis=0)
