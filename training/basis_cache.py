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


def _key_raw(npz_dir, nx, ny, nt, x_end, y_end, drop_first_steps,
             seed, train_frac, val_frac, n_fit) -> str:
    """LEGACY key formula (kept for backward compatibility).

    Same as _key but does NOT resolve npz_dir. Any pod3d_*.npz file
    written by pre-normalize versions of the code lives under this
    hash. load_or_fit_basis checks this as a fallback so an existing
    cache does not become dead just because we changed the key
    formula. Do not call this from new code; call _key which
    normalizes npz_dir first.

    Also picks up the loader's current rim-mask r_end so any change
    to _DISK_MASK_R_END auto-invalidates the basis cache (fitting a
    basis on data that used the OLD rim mask would produce different
    modes; we must NOT HIT the old file after the tighten)."""
    from data.loader import _DISK_MASK_R_END
    raw = (f"pod3d|{npz_dir}|{nx}|{ny}|{nt}|{x_end}|{y_end}|"
           f"{drop_first_steps}|{seed}|{train_frac}|{val_frac}|"
           f"{n_fit}|rim={_DISK_MASK_R_END}")
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _key(npz_dir, nx, ny, nt, x_end, y_end, drop_first_steps,
         seed, train_frac, val_frac, n_fit) -> str:
    """Stable short hash that identifies the data-split-grid context.

    npz_dir is resolved to canonical absolute form FIRST so different
    string spellings that map to the same physical directory (trailing
    slash, symlink, ~, relative-vs-absolute) all produce the same hash.
    This prevents a common source of cache MISS: training was run with
    '--data.npz_dir /path/to/data/' (trailing slash) but viz_all with
    '--npz-dir /path/to/data' (no slash), or one path via symlink and
    the other via the target.

    Every other parameter that changes the contents of the (train +
    val) sims goes into the raw string as-is:
      - nx, ny, nt, x_end, y_end: canonical grid
      - drop_first_steps: loader trim, alters every sim's f
      - seed + train_frac + val_frac: which sims end up in (train + val)
      - n_fit: count of sims actually used (catches limit-changes)

    Falls back to the raw string if resolve() fails (rare; only when
    the path is invalid, which load_or_fit_basis would have already
    caught during load_dataset).
    """
    try:
        npz_dir_norm = str(
            Path(str(npz_dir)).expanduser().resolve())
    except (OSError, ValueError):
        npz_dir_norm = str(npz_dir)
    return _key_raw(npz_dir_norm, nx, ny, nt, x_end, y_end,
                     drop_first_steps, seed, train_frac,
                     val_frac, n_fit)


def _make(Phi, sigma, spatial_shape) -> PODBasis:
    return PODBasis(Phi=Phi.copy(), sigma=sigma.copy(),
                    spatial_shape=tuple(int(d) for d in spatial_shape))


def load_cached_basis(*, npz_dir, nx, ny, nt, x_end, y_end,
                       drop_first_steps, seed, train_frac, val_frac,
                       n_fit, cache_dir, K: int) -> PODBasis | None:
    """Load an existing pod3d cache file without needing the sims
    list. Returns None if no compatible cache exists. Tries the
    resolved-npz_dir key first, then falls back to the legacy raw
    key (mirrors load_or_fit_basis's fallback ordering)."""
    cache_dir = Path(cache_dir)
    resolved_key = _key(npz_dir, nx, ny, nt, x_end, y_end,
                          drop_first_steps, seed, train_frac,
                          val_frac, n_fit)
    legacy_key = _key_raw(npz_dir, nx, ny, nt, x_end, y_end,
                           drop_first_steps, seed, train_frac,
                           val_frac, n_fit)
    for key in (resolved_key, legacy_key):
        path = cache_dir / f"pod3d_{key}.npz"
        if not path.exists():
            continue
        try:
            with np.load(path, allow_pickle=False) as z:
                if int(z["k_cache"]) >= K:
                    return _make(z["Phi"][:, :K], z["sigma"][:K],
                                  z["spatial_shape"])
        except (OSError, KeyError, ValueError):
            continue
    return None


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

    # Build the list of paths to try, in preference order:
    #   1. resolved-key path (the canonical location under the new
    #      key formula that normalizes npz_dir)
    #   2. legacy-key path (raw npz_dir; only present if a pre-fix
    #      version wrote it, e.g. this user's existing pod3d files
    #      from before Path.resolve() landed in _key). Skipped when
    #      the raw string already resolves to the same value, since
    #      then legacy_key == resolved_key and the file is the same.
    candidates = [(path, "resolved-key")]
    legacy_key = _key_raw(npz_dir, nx, ny, nt, x_end, y_end,
                           drop_first_steps, seed, train_frac,
                           val_frac, n_fit)
    if legacy_key != key:
        legacy_path = cache_dir / f"pod3d_{legacy_key}.npz"
        candidates.append((legacy_path, "legacy-key"))

    if not force_refit:
        for cand_path, label in candidates:
            if not cand_path.exists():
                continue
            try:
                with np.load(cand_path, allow_pickle=False) as z:
                    if int(z["k_cache"]) >= K:
                        print(f"  POD basis cache HIT ({label}): "
                              f"{cand_path.name} (k_cache="
                              f"{int(z['k_cache'])}, asked K={K})",
                              flush=True)
                        return _make(z["Phi"][:, :K], z["sigma"][:K],
                                     z["spatial_shape"])
                    else:
                        print(f"  POD basis cache stale ({label}): "
                              f"k_cache={int(z['k_cache'])} < "
                              f"requested K={K}; refitting at k_cache="
                              f"{k_cache}", flush=True)
                        break   # will refit; stop probing candidates
            except (OSError, KeyError, ValueError) as e:
                print(f"  POD basis cache unreadable ({label}: "
                      f"{cand_path.name}, {e}); trying next",
                      flush=True)

    # Diagnostic MISS logging: show the key tuple the caller resolved
    # and list any existing pod3d_*.npz that might have had a different
    # key. Helps operators see WHICH field differed when they thought
    # the cache should have hit (e.g. training vs viz using different
    # npz_dir string, or an unnoticed change in train_frac / seed).
    print(f"  POD basis cache MISS: expected {path.name}", flush=True)
    print(f"    key tuple: npz_dir={npz_dir!r} nx={nx} ny={ny} nt={nt} "
          f"x_end={x_end} y_end={y_end} "
          f"drop_first_steps={drop_first_steps} seed={seed} "
          f"train_frac={train_frac} val_frac={val_frac} n_fit={n_fit}",
          flush=True)
    _existing = sorted(cache_dir.glob("pod3d_*.npz"))
    if _existing:
        print(f"    existing pod3d files under {cache_dir}:", flush=True)
        for _p in _existing[:8]:
            print(f"      {_p.name}", flush=True)
        if len(_existing) > 8:
            print(f"      ... and {len(_existing) - 8} more", flush=True)
        print(f"    if one of these was your training's basis, some key "
              f"element above differs from what training saw. Grep "
              f"outputs/<tag>/results.json for the ones you can inspect "
              f"(npz_dir, data.seed, train.val_frac, data.train_frac, "
              f"data.drop_first_steps).",
              flush=True)
    print(f"  fitting K={k_cache} on {n_fit} sims "
          f"(workers={workers}) ...", flush=True)
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
