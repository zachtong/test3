"""Core data structure for a single 3D simulation.

The field `f` lives on the canonical Cartesian grid (Nx, Ny, Nt). No
axisymmetric front (`rb`) is recorded -- in the lab-frame full-3D picture
there is no scalar `r_b(t)`, the contact front is a contour rb(theta, t) /
level set, and current POD reconstruction does not need it.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Simulation:
    f: np.ndarray       # (Nx, Ny, Nt) gap field on the canonical grid
    params: Any = None  # simulation parameters (varies by source)
