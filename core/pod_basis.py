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
import time
import numpy as np

from core.simulation import Simulation


def _accumulate_gram_serial(sims: list, n_space: int,
                            verbose: bool) -> np.ndarray:
    """Serial streaming Gram with a pre-allocated tmp buffer.

    Without the tmp buffer, every iteration's `F @ F.T` allocates a
    fresh (n_space, n_space) float64 -- 2 GB at firehorse2 scale --
    which malloc cannot satisfy from a free-list and thrashes the
    page cache. With np.dot(out=tmp), the buffer is allocated once and
    reused, dropping per-sim wall time roughly 2-3x in practice.
    """
    C = np.zeros((n_space, n_space), dtype=np.float64)
    tmp = np.empty((n_space, n_space), dtype=np.float64)
    n_sims = len(sims)
    t0 = time.time()
    last = [0.0]
    for i, s in enumerate(sims, 1):
        F = np.asarray(s.f, dtype=np.float64).reshape(n_space, -1)
        np.dot(F, F.T, out=tmp)
        C += tmp
        now = time.time()
        if verbose and (now - last[0] >= 5.0 or i == n_sims):
            rate = i / max(now - t0, 1e-9)
            eta_min = (n_sims - i) / max(rate, 1e-9) / 60.0
            print(f"  POD streaming Gram (serial): {i}/{n_sims} sim "
                  f"({100.0 * i / n_sims:.1f}%)  "
                  f"{rate:.1f}/s  ETA {eta_min:.1f} min", flush=True)
            last[0] = now
    return C


def _gram_chunk_worker(args) -> np.ndarray:
    """Compute the partial Gram for a chunk of sims; top-level for
    ProcessPoolExecutor picklability."""
    sims_chunk, n_space = args
    Cp = np.zeros((n_space, n_space), dtype=np.float64)
    tmp = np.empty((n_space, n_space), dtype=np.float64)
    for s in sims_chunk:
        F = np.asarray(s.f, dtype=np.float64).reshape(n_space, -1)
        np.dot(F, F.T, out=tmp)
        Cp += tmp
    return Cp


def _accumulate_gram_parallel(sims: list, n_space: int, workers: int,
                              verbose: bool) -> np.ndarray:
    """Partial-Gram parallel reduce.

    Sims are split into `workers` round-robin chunks (round-robin
    rather than contiguous, so any temporal trend in sim ordering
    spreads evenly across workers and per-worker work is balanced).
    Each worker carries one extra (n_space, n_space) float64 Gram
    (~2 GB at firehorse2 scale); the main process accumulates each
    returned partial into a single C as it arrives.

    IPC cost: each worker pickles + returns a 2 GB float64 array. On
    Linux fork that's the only IPC, and at 32 workers the deserialize
    + add takes seconds vs the minutes saved by parallel GEMM.
    """
    from concurrent.futures import ProcessPoolExecutor
    chunks = [sims[i::workers] for i in range(workers)]
    chunks = [c for c in chunks if c]
    workers = len(chunks)
    if verbose:
        print(f"  POD streaming Gram (parallel, {workers} workers): "
              f"{len(sims)} sim across "
              f"{[len(c) for c in chunks][:5]}{'...' if workers > 5 else ''} "
              f"per-worker batches", flush=True)
    C = np.zeros((n_space, n_space), dtype=np.float64)
    t0 = time.time()
    done = [0]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for Cp in ex.map(_gram_chunk_worker,
                         [(c, n_space) for c in chunks]):
            done[0] += 1
            C += Cp
            if verbose:
                rate = done[0] / max(time.time() - t0, 1e-9)
                eta_min = ((workers - done[0]) / max(rate, 1e-9)
                           / 60.0)
                print(f"  POD partial Gram: {done[0]}/{workers} "
                      f"chunks merged  ETA {eta_min:.1f} min",
                      flush=True)
    return C


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
            verbose: bool = True,
            workers: int | None = None) -> "PODBasis":
        """Fit POD via streaming method of snapshots.

        C = sum_i F_i F_i^T is accumulated one simulation at a time, so
        peak RAM is the spatial Gram (Nx*Ny, Nx*Ny) float64 plus one
        per-sim slice (Nx*Ny, Nt) float64 -- never the whole snapshot
        matrix. Identical algorithm to the 2D PODBasis.fit; the
        comments call it 'streaming' here because at 3D resolutions the
        single-shot version actually matters.

        Two perf wins vs the naive write of this loop:

          1. Pre-allocate a (Nx*Ny, Nx*Ny) float64 tmp buffer for the
             per-sim outer product and use np.dot(F, F.T, out=tmp). The
             naive `C += F @ F.T` allocates a fresh 2 GB matrix every
             iteration (3D firehorse2 scale), thrashing malloc and
             killing wall time.
          2. Optional `workers` enables a partial-Gram parallel reduce:
             each worker streams a contiguous chunk of sims into its
             own Gram, the main process sums them at the end. At
             firehorse2 scale (n_space = 16384) a single sim's GEMM is
             ~160 GFLOPs -- saturating a single BLAS thread is wasteful
             when 32+ cores are sitting idle. Each worker carries one
             extra 2 GB Gram, so 32 workers cost ~64 GB peak (fine on a
             1 TB workstation). workers=None defaults to serial; set
             explicitly to enable parallel fit.

        Args:
            sims: iterable of Simulation; each `sim.f` is (Nx, Ny, Nt)
                  with the same (Nx, Ny). Iteration order is not
                  significant -- C is order-invariant.
            K: number of leading POD modes to return.
            verbose: print one progress line every 5 s during the
                  accumulation loop (set False for unit tests).
            workers: parallel Gram-accumulation worker count. None /
                  0 / 1 = serial. On a fork-based platform (Linux)
                  sims are inherited copy-on-write, so passing the full
                  list to workers is free; on spawn-based platforms
                  (macOS / Windows) each worker re-pickles its chunk's
                  sims, which adds IPC cost.
        """
        import time
        sims = list(sims)
        if not sims:
            raise ValueError("PODBasis.fit: empty sim list")
        nx, ny, _ = sims[0].f.shape
        n_space = nx * ny

        t0 = time.time()
        if workers and workers > 1 and len(sims) >= workers:
            C = _accumulate_gram_parallel(sims, n_space, workers, verbose)
        else:
            C = _accumulate_gram_serial(sims, n_space, verbose)

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
        """Return a (K, Nt): lab-frame projection of one sim's field.

        Perf note: we intentionally do NOT do astype(float64) on
        sim.f -- that would allocate a 39 MB temp per call (128*128
        *300 * 8 bytes), which is memory-bandwidth-bound and stalls
        for thousands of sims. numpy.matmul upcasts on the fly when
        one input is float64 (Phi) and the other is float32 (sim.f),
        without producing a full-size intermediate.
        """
        nx, ny = self.spatial_shape
        F = sim.f.reshape(nx * ny, -1)          # native dtype, no copy
        return self.Phi.T @ F

    def project_ensemble(self, sims,
                          progress_every: int = 500) -> np.ndarray:
        """Stack to (N_sim, K, Nt). Prints a heartbeat every
        `progress_every` sims so long ensembles are diagnosable."""
        import time
        sims = list(sims)
        n = len(sims)
        out = []
        t0 = time.time()
        last = t0
        for i, s in enumerate(sims, 1):
            out.append(self.project_sim(s))
            now = time.time()
            if progress_every and (i % progress_every == 0 or i == n):
                rate = i / max(now - t0, 1e-9)
                eta = (n - i) / max(rate, 1e-9)
                print(f"  project_ensemble: {i}/{n}  "
                      f"({rate:.1f}/s  ETA {eta:.1f}s)", flush=True)
                last = now
        return np.stack(out, axis=0)

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
