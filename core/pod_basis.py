"""Plain POD basis over a 3D Cartesian field (Nx, Ny, Nt).

POD is just SVD on the snapshot matrix; the spatial dimensionality is
absorbed by flattening (Nx, Ny) -> Nx*Ny. The interface deliberately mirrors
the 2D `wafer_bonding_sparse_recon` `PODBasis` so downstream training and
evaluation code (dataset builder, scorer, baselines, bundle packager) ports
without case-splits on dimensionality:

  fit / project_sim / project_ensemble / reconstruct / energy_fraction
  uses_front = False  (no co-moving shift; no front channel)

Implementation: method of snapshots on the (Nx*Ny, Nx*Ny) covariance, so
memory is independent of the simulation count. The wafer-disk mask (off-disk
cells) is currently NOT enforced here -- callers can zero the off-disk
region before passing sims in, or pass it in via a future `mask=` kwarg.
"""

from __future__ import annotations
from typing import Iterable
import numpy as np

from core.simulation import Simulation


class PODBasis:
    uses_front = False    # no moving-frame shift; reconstruction needs no r_b

    def __init__(self, Phi: np.ndarray, sigma: np.ndarray,
                 spatial_shape: tuple[int, int]) -> None:
        # Phi: (Nx*Ny, K) orthonormal flattened spatial modes
        # sigma: (K,) singular values
        # spatial_shape: (Nx, Ny) so reconstruct() can unflatten back
        self.Phi = Phi
        self.sigma = sigma
        self.spatial_shape = spatial_shape

    @classmethod
    def fit(cls, sims: Iterable[Simulation], K: int) -> "PODBasis":
        """Fit POD on the snapshot matrix; pick the smaller Gram side.

        Standard SVD identity: phi_k can come from either
          (a) eig of the SPATIAL Gram  X X^T, size (n_space, n_space)
          (b) eig of the SNAPSHOT Gram X^T X, size (n_snap, n_snap),
              then Phi = X V / sigma  (Sirovich's method of snapshots).

        2D wafer_bonding_sparse_recon used (a) because n_space ~= 1000.
        In 3D n_space = Nx*Ny easily hits 10^4-10^5 so the (a) Gram and its
        O(n_space^3) eigh blow up; switch to whichever side is smaller.

        Note: assumes the snapshot matrix X = [F_1 | F_2 | ...] fits in RAM
        once (no streaming). For 128x128, Nt=300, 500 sims that's
        128*128*300*500*8 B ~= 24 GB -- on a workstation this is fine but
        not on a laptop. A streaming/randomized variant is future work.
        """
        sims = list(sims)
        if not sims:
            raise ValueError("PODBasis.fit: empty sim list")
        nx, ny, _ = sims[0].f.shape
        n_space = nx * ny

        cols = [np.asarray(s.f, dtype=np.float64).reshape(n_space, -1)
                for s in sims]
        X = np.concatenate(cols, axis=1)                 # (n_space, n_snap)
        n_snap = X.shape[1]

        if n_snap <= n_space:
            # Snapshot-side Gram is smaller -- standard method of snapshots.
            G = X.T @ X                                   # (n_snap, n_snap)
            w, V = np.linalg.eigh(G)
            order = np.argsort(w)[::-1][:K]
            sigma2 = np.clip(w[order], 0.0, None)
            sigma = np.sqrt(sigma2)
            V_k = V[:, order]
            Phi = np.zeros((n_space, K), dtype=np.float64)
            nz = sigma > 1e-12
            Phi[:, nz] = (X @ V_k[:, nz]) / sigma[nz]
        else:
            # Spatial-side Gram is smaller -- match the 2D codepath exactly.
            C = X @ X.T                                   # (n_space, n_space)
            w, V = np.linalg.eigh(C)
            order = np.argsort(w)[::-1][:K]
            Phi = np.ascontiguousarray(V[:, order])
            sigma = np.sqrt(np.clip(w[order], 0.0, None))
        return cls(Phi=Phi, sigma=sigma, spatial_shape=(nx, ny))

    def project_sim(self, sim: Simulation) -> np.ndarray:
        """Return a (K, Nt): lab-frame projection of one sim's field."""
        nx, ny = self.spatial_shape
        F = np.asarray(sim.f, dtype=np.float64).reshape(nx * ny, -1)
        return self.Phi.T @ F

    def project_ensemble(self, sims: Iterable[Simulation]) -> np.ndarray:
        """Stack to (N_sim, K, Nt)."""
        return np.stack([self.project_sim(s) for s in sims], axis=0)

    def reconstruct(self, a: np.ndarray) -> np.ndarray:
        """Inverse map a(K, Nt) -> f(Nx, Ny, Nt). `rb` is intentionally absent
        from the signature (no co-moving shift); a stray rb argument from a
        2D-style call site would be a quiet bug, so don't accept one."""
        nx, ny = self.spatial_shape
        F = self.Phi @ a
        return F.reshape(nx, ny, -1)

    def energy_fraction(self) -> np.ndarray:
        """Cumulative energy captured by the leading k modes (k=1..K)."""
        e = self.sigma ** 2
        return np.cumsum(e) / e.sum()
