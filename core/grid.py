"""Canonical Cartesian grid (quarter-disk) and polar <-> Cartesian helpers.

The simulation enforces 90-degree rotational symmetry about the z axis, so
the data lives entirely in the first quadrant. The canonical grid covers
`x in [0, x_end]` and `y in [0, y_end]` (both normalized by the wafer
radius R; the wafer's first-quadrant arc is the curve x^2 + y^2 = 1 when
x_end = y_end = 1.0). Off-disk and off-quadrant cells are masked out by
`disk_mask` so downstream code can zero them before field-error or POD math.

Sensor placements are specified in polar (r, theta) in the config because
that matches the rig-side intuition ("an edge sensor on the X axis at
r = 1.0"); theta is constrained to [0, 90] degrees.
"""

from __future__ import annotations

import numpy as np


def canonical_grid(nx: int, ny: int, x_end: float = 1.0, y_end: float = 1.0
                   ) -> tuple[np.ndarray, np.ndarray]:
    """Return (x_canon, y_canon) -- 1D coord vectors, length nx and ny.

    Quarter layout: both axes run from 0 to the corresponding `*_end`. The
    wafer's first-quadrant region is the quarter-disk `x^2 + y^2 <= 1` when
    x_end = y_end = 1.0.
    """
    x = np.linspace(0.0, x_end, nx)
    y = np.linspace(0.0, y_end, ny)
    return x, y


def disk_mask(nx: int, ny: int, x_end: float = 1.0, y_end: float = 1.0,
              r_end: float = 1.0) -> np.ndarray:
    """Boolean (Nx, Ny) mask: True where the cell is INSIDE the quarter-disk.

    A cell is in-disk iff x >= 0, y >= 0 (the first-quadrant constraint is
    implicit in `canonical_grid` but stated here for clarity), and
    x^2 + y^2 <= r_end^2.
    """
    x, y = canonical_grid(nx, ny, x_end, y_end)
    X, Y = np.meshgrid(x, y, indexing="ij")
    return (X >= 0.0) & (Y >= 0.0) & (X * X + Y * Y <= r_end * r_end)


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
