"""Linear sparse-sensing baselines for the 3D gap field.

STUB. The 2D codebase ships a Gappy POD baseline (Carlberg / Willcox style)
in this file: given a fitted POD basis Phi (Nx, K), n sensor rows S, and a
sensor reading vector y (n,), solve the least-squares
    a_hat = argmin || S Phi a - y ||
field is then Phi a_hat. With K <= n this is a well-posed N-of-n problem;
with K > n it has to be regularized or it under-determines.

In 3D the math is the same -- flatten (Nx, Ny) -> Nx*Ny so Phi is
(Nx*Ny, K), the n sensor rows are the (Nx*Ny)-indices for each sensor's
flattened position, and the solve / reconstruction proceed unchanged. The
file is left empty for now because the surrounding flow doesn't need a
baseline until results comparison runs.
"""

from __future__ import annotations
import numpy as np

from core.pod_basis import PODBasis


def gappy_pod(basis: PODBasis, sensor_flat_idx: np.ndarray,
              y_sensor: np.ndarray) -> np.ndarray:
    """STUB: predict field coefficients a from sensor readings.

    Args:
        basis: fitted PODBasis (Phi (Nx*Ny, K), sigma (K,)).
        sensor_flat_idx: (n,) integer indices into the flattened spatial
            axis (Nx*Ny) for the n sensors.
        y_sensor: (n, Nt) sensor traces.

    Returns:
        a: (K, Nt) least-squares POD coefficients.
    """
    raise NotImplementedError(
        "evaluation/baselines.py::gappy_pod: deferred until first "
        "comparison run. Trivial to fill from 2D `evaluation/baselines.py`.")
