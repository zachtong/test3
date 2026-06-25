"""Canonical Cartesian grid and polar <-> Cartesian helpers.

The canonical grid is x in [-x_end, x_end] and y in [-y_end, y_end] (both
normalized by the wafer radius R; the wafer is a unit disk in normalized
coords when x_end = y_end = 1.0). Off-disk grid cells (r > 1.0) hold the
sentinel value `OFF_WAFER` so downstream code can mask them out before
field-error or projection math.

Sensor placements are specified in polar (r, theta) in the config because
that matches the rig-side intuition ("an edge sensor on the X axis at
r = 1.0"); index lookup is done against this Cartesian grid.
"""

from __future__ import annotations

import numpy as np


OFF_WAFER = np.float64("nan")


def canonical_grid(nx: int, ny: int, x_end: float = 1.0, y_end: float = 1.0
                   ) -> tuple[np.ndarray, np.ndarray]:
    """Return (x_canon, y_canon) -- 1D coord vectors, length nx and ny.

    Grids are centred on the origin so the wafer disk lies symmetrically.
    """
    x = np.linspace(-x_end, x_end, nx)
    y = np.linspace(-y_end, y_end, ny)
    return x, y


def disk_mask(nx: int, ny: int, x_end: float = 1.0, y_end: float = 1.0,
              r_end: float = 1.0) -> np.ndarray:
    """Boolean (Nx, Ny) mask: True where the cell is INSIDE the wafer disk.

    Use to zero/ignore off-wafer cells in field-error / POD math; the wafer
    is a circle of normalized radius `r_end` (default 1.0).
    """
    x, y = canonical_grid(nx, ny, x_end, y_end)
    X, Y = np.meshgrid(x, y, indexing="ij")
    return X * X + Y * Y <= r_end * r_end


def polar_to_xy(r: float, theta_deg: float) -> tuple[float, float]:
    """(r, theta_deg) -> (x, y) in the same normalized units."""
    t = np.deg2rad(theta_deg)
    return float(r * np.cos(t)), float(r * np.sin(t))


def xy_to_indices(x: float, y: float, x_canon: np.ndarray,
                  y_canon: np.ndarray) -> tuple[int, int]:
    """Nearest (ix, iy) on the canonical grid for a Cartesian point."""
    ix = int(np.argmin(np.abs(x_canon - x)))
    iy = int(np.argmin(np.abs(y_canon - y)))
    return ix, iy
