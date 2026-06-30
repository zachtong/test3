"""Plain POD basis over a 3D Cartesian field (Nx, Ny, Nt).

POD is the SVD of the snapshot matrix X of shape (Nx*Ny, N_sim*Nt). For
the firehorse2-scale data (Nx=Ny=128, Nt=300, N_sim=5500) X is 108 GB --
too big to assemble even on a 1 TB workstation when intermediate GEMM
buffers are counted. fit() therefore uses the streaming method-of-
snapshots: accumulate the spatial Gram C = sum_i F_i F_i^T one
simulation at a time (C is only (Nx*Ny, Nx*Ny) float64 = 2 GB for
Nx=128), then eigh(C) for the top-K modes. RAM peak ~2.5 GB regardless
of how many sims are passed in.

Interface mirrors the 2D `wafer_bonding_sparse_recon` PODBasis so
downstream training and evaluation code (dataset builder, scorer,
baselines, bundle packager) ports without case-splits on dimensionality:

  fit / project_sim / project_ensemble / reconstruct / energy_fraction
  uses_front = False  (no co-moving shift; no front channel)

The wafer-disk mask (off-disk cells) is not enforced here -- the loader
already zeroes them in sim.f before fit() sees them.
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
    def fit(cls, sims: Iterable[Simulation], K: int,
            verbose: bool = True) -> "PODBasis":
        """Fit POD via streaming method of snapshots.

        C = sum_i F_i F_i^T is accumulated one simulation at a time, so
        peak RAM is the spatial Gram (Nx*Ny, Nx*Ny) float64 plus one
        per-sim slice (Nx*Ny, Nt) float64 -- never the whole snapshot
        matrix. Identical algorithm to the 2D PODBasis.fit; the
        comments call it 'streaming' here because at 3D resolutions the
        single-shot version actually matters.

        Args:
            sims: iterable of Simulation; each `sim.f` is (Nx, Ny, Nt)
                  with the same (Nx, Ny). Iteration order is not
                  significant -- C is order-invariant.
            K: number of leading POD modes to return.
            verbose: print one progress line every 5 s during the
                  accumulation loop (set False for unit tests).
        """
        import time
        sims = list(sims)
        if not sims:
            raise ValueError("PODBasis.fit: empty sim list")
        nx, ny, _ = sims[0].f.shape
        n_space = nx * ny

        # Streaming accumulate the spatial Gram. float64 to keep
        # eigh well-conditioned -- the per-sim outer products are tiny
        # numbers (displacement ~1e-4 m) so single precision would
        # underflow on a 5500-sim sum.
        C = np.zeros((n_space, n_space), dtype=np.float64)
        n_sims = len(sims)
        t0 = time.time()
        last = [0.0]
        for i, s in enumerate(sims, 1):
            F = np.asarray(s.f, dtype=np.float64).reshape(n_space, -1)
            C += F @ F.T
            now = time.time()
            if verbose and (now - last[0] >= 5.0 or i == n_sims):
                rate = i / max(now - t0, 1e-9)
                eta_min = (n_sims - i) / max(rate, 1e-9) / 60.0
                print(f"  POD streaming Gram: {i}/{n_sims} sim "
                      f"({100.0 * i / n_sims:.1f}%)  "
                      f"{rate:.1f}/s  ETA {eta_min:.1f} min",
                      flush=True)
                last[0] = now

        if verbose:
            print(f"  POD eigh on ({n_space}, {n_space}) ...", flush=True)
        t_eigh = time.time()
        w, V = np.linalg.eigh(C)                         # ascending
        order = np.argsort(w)[::-1][:K]
        Phi = np.ascontiguousarray(V[:, order])
        sigma = np.sqrt(np.clip(w[order], 0.0, None))
        if verbose:
            print(f"  POD eigh: {time.time() - t_eigh:.0f}s  "
                  f"total fit: {(time.time() - t0) / 60.0:.1f} min",
                  flush=True)
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
