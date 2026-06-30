"""Disk cache for fitted POD bases so training runs skip the SVD.

A basis is fit on the (train + val) split, so the cache key captures
the data source, grid, drop policy, and split parameters. Because POD
modes are nested in K (the top-k columns of Phi are exactly the same
as if you had asked for k modes from scratch), one cache file with a
larger k_cache satisfies every K <= k_cache request -- fit once at
K=32, reuse for K=4 / 8 / 16 / ... without re-running eigh.

Bases live in one shared directory (NOT under any run's tag folder),
keyed by a hash of the identifying parameters, so different
experiments on the same data / split share a single fit.

Cache miss: fit the basis at max(K, k_cache) and write the file.
Cache hit and k_cache_on_disk >= K: slice Phi[:, :K] + sigma[:K] and
return without touching the data.
"""

from __future__ import annotations
from pathlib import Path
import hashlib
import time

import numpy as np

from core.pod_basis import PODBasis


def _key(npz_dir, nx, ny, nt, x_end, y_end, drop_first_steps,
         seed, train_frac, val_frac, n_fit) -> str:
    """Stable short hash that identifies the data-split-grid context.

    Every parameter that changes the contents of the (train + val) sims
    must be in the raw string. Notably:
      - npz_dir: the data folder
      - nx, ny, nt, x_end, y_end: canonical grid
      - drop_first_steps: loader trim, alters every sim's f
      - seed + train_frac + val_frac: which sims end up in (train + val)
      - n_fit: count of sims actually used (catches limit-changes)
    """
    raw = (f"pod3d|{npz_dir}|{nx}|{ny}|{nt}|{x_end}|{y_end}|"
           f"{drop_first_steps}|{seed}|{train_frac}|{val_frac}|{n_fit}")
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _make(Phi, sigma, spatial_shape) -> PODBasis:
    return PODBasis(Phi=Phi.copy(), sigma=sigma.copy(),
                    spatial_shape=tuple(int(d) for d in spatial_shape))


def load_or_fit_basis(sims, K: int, *, npz_dir, nx, ny, nt,
                      x_end, y_end, drop_first_steps,
                      seed, train_frac, val_frac,
                      cache_dir, k_cache: int = 16,
                      force_refit: bool = False,
                      workers: int | None = None) -> PODBasis:
    """Return a K-mode PODBasis, loading from / saving to a shared cache.

    Args:
        sims: the (train + val) Simulation list to fit on if the cache
              misses; never read when the cache hits.
        K: number of leading POD modes the caller wants right now.
        npz_dir, nx, ny, nt, x_end, y_end, drop_first_steps,
        seed, train_frac, val_frac: feed into the cache key so the
              same (data, split, grid, trim) shares one fit.
        cache_dir: shared directory (not per-tag); created if missing.
        k_cache: the K used for the underlying fit -- bigger means more
              cached headroom for future requests at smaller K, at the
              cost of marginally more eigh work (eigh always computes
              all eigenvalues; the slice is just storage). Default 16.
        force_refit: ignore any cache hit and refit. Useful when the
              user knows something off-key changed (e.g. raw NPZ data
              was edited in place) and wants to force a rebuild.

    Returns a PODBasis with exactly K modes.
    """
    if K > k_cache:
        # Honour an oversized request by raising k_cache so we store
        # enough columns to satisfy this and future calls.
        k_cache = K
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    n_fit = len(sims)
    key = _key(npz_dir, nx, ny, nt, x_end, y_end, drop_first_steps,
               seed, train_frac, val_frac, n_fit)
    path = cache_dir / f"pod3d_{key}.npz"

    if not force_refit and path.exists():
        try:
            with np.load(path, allow_pickle=False) as z:
                if int(z["k_cache"]) >= K:
                    print(f"  POD basis cache HIT: {path.name} "
                          f"(k_cache={int(z['k_cache'])}, asked K={K})",
                          flush=True)
                    return _make(z["Phi"][:, :K], z["sigma"][:K],
                                 z["spatial_shape"])
                else:
                    print(f"  POD basis cache stale: k_cache="
                          f"{int(z['k_cache'])} < requested K={K}; "
                          f"refitting at k_cache={k_cache}", flush=True)
        except (OSError, KeyError, ValueError) as e:
            print(f"  POD basis cache unreadable ({e}); refitting",
                  flush=True)

    print(f"  POD basis cache MISS: fitting K={k_cache} on "
          f"{n_fit} sims (workers={workers}) ...", flush=True)
    t0 = time.time()
    full = PODBasis.fit(sims, K=k_cache, workers=workers)
    print(f"  POD basis fit in {(time.time() - t0) / 60.0:.1f} min",
          flush=True)
    # Atomic write so a kill during save doesn't leave a corrupt file
    # that the next run would (correctly) reject as unreadable.
    tmp = path.with_suffix(".tmp.npz")
    np.savez(tmp, Phi=full.Phi, sigma=full.sigma,
             spatial_shape=np.asarray(full.spatial_shape, dtype=np.int64),
             k_cache=np.asarray(full.Phi.shape[1], dtype=np.int64))
    tmp.replace(path)
    print(f"  POD basis cached -> {path.name}", flush=True)
    return _make(full.Phi[:, :K], full.sigma[:K], full.spatial_shape)
