"""Tests for the POD basis disk cache."""

from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pytest

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from core.simulation import Simulation                       # noqa: E402
from core.pod_basis import PODBasis                          # noqa: E402
from training.basis_cache import (load_or_fit_basis,         # noqa: E402
                                    _key, _key_raw)


def _mock_sims(n_sim: int = 6, nx: int = 16, ny: int = 16,
               nt: int = 8, seed: int = 0):
    """A few small Simulation objects with random fields.

    Random data is fine here -- we are testing cache mechanics
    (hit / miss / slice / refit), not POD correctness.
    """
    rng = np.random.default_rng(seed)
    sims = []
    for _ in range(n_sim):
        f = rng.standard_normal((nx, ny, nt)).astype(np.float32)
        sims.append(Simulation(f=f, params={}))
    return sims


def _kwargs(cache_dir: Path) -> dict:
    """Identifying kwargs for the cache key; shared across all tests."""
    return dict(npz_dir="/fake/path", nx=16, ny=16, nt=8,
                x_end=1.0, y_end=1.0, drop_first_steps=0,
                seed=7, train_frac=0.8, val_frac=0.1,
                cache_dir=cache_dir)


def test_first_call_writes_cache_file(tmp_path):
    sims = _mock_sims()
    basis = load_or_fit_basis(sims, K=4, **_kwargs(tmp_path), k_cache=8)
    assert isinstance(basis, PODBasis)
    assert basis.Phi.shape == (16 * 16, 4)
    assert basis.sigma.shape == (4,)
    # Exactly one cache file should now exist.
    cached = list(tmp_path.glob("pod3d_*.npz"))
    assert len(cached) == 1


def test_second_call_hits_cache_without_fit(tmp_path):
    """If the cache exists and k_cache_on_disk >= K, the SVD must not
    re-run. We detect 'no refit' by checking that fit() isn't called --
    the cache file's contents would otherwise change."""
    sims = _mock_sims()
    basis_a = load_or_fit_basis(sims, K=4, **_kwargs(tmp_path), k_cache=8)
    cached = next(tmp_path.glob("pod3d_*.npz"))
    mtime0 = cached.stat().st_mtime

    # Smaller K than k_cache should be a hit and slice.
    basis_b = load_or_fit_basis(sims, K=2, **_kwargs(tmp_path), k_cache=8)
    assert basis_b.Phi.shape == (16 * 16, 2)
    assert np.allclose(basis_b.Phi, basis_a.Phi[:, :2])
    assert np.allclose(basis_b.sigma, basis_a.sigma[:2])
    assert cached.stat().st_mtime == mtime0, "cache was rewritten on hit"


def test_K_above_k_cache_refits_and_raises_k_cache(tmp_path):
    """Asking for K > k_cache_on_disk should refit at the new K."""
    sims = _mock_sims()
    load_or_fit_basis(sims, K=4, **_kwargs(tmp_path), k_cache=4)
    cached = next(tmp_path.glob("pod3d_*.npz"))
    with np.load(cached) as z:
        assert int(z["k_cache"]) == 4
    # Demand K=6 > k_cache=4 -> refit at K=6 (k_cache auto-bumped).
    basis = load_or_fit_basis(sims, K=6, **_kwargs(tmp_path), k_cache=4)
    assert basis.Phi.shape == (16 * 16, 6)
    with np.load(cached) as z:
        assert int(z["k_cache"]) == 6


def test_force_refit_ignores_cache(tmp_path):
    sims = _mock_sims()
    load_or_fit_basis(sims, K=4, **_kwargs(tmp_path), k_cache=8)
    cached = next(tmp_path.glob("pod3d_*.npz"))
    mtime0 = cached.stat().st_mtime
    # Tiny sleep would be cleaner but Path.touch tolerance is fine.
    import os
    os.utime(cached, (mtime0 - 100, mtime0 - 100))
    load_or_fit_basis(sims, K=4, **_kwargs(tmp_path), k_cache=8,
                      force_refit=True)
    assert cached.stat().st_mtime > mtime0 - 100, "force_refit did not rewrite"


def test_trailing_slash_and_expanduser_normalize_to_same_key(tmp_path):
    """Same physical directory expressed differently (trailing slash,
    'ok' vs 'ok/', relative-then-resolved-equivalent) must produce the
    SAME resolved cache key. This is the whole point of the resolve()
    call in _key; without it, training with --npz-dir 'foo/' and viz
    with --npz-dir 'foo' would each fit a separate basis for the
    same data."""
    real = tmp_path / "real_data"
    real.mkdir()
    common = dict(nx=16, ny=16, nt=8, x_end=1.0, y_end=1.0,
                   drop_first_steps=0, seed=7, train_frac=0.8,
                   val_frac=0.1, n_fit=6)
    k_no_slash = _key(str(real),      **common)
    k_slash    = _key(str(real) + "/", **common)
    # Also test '.' relative variant
    import os
    prev = os.getcwd(); os.chdir(tmp_path)
    try:
        k_rel = _key("real_data", **common)
    finally:
        os.chdir(prev)
    assert k_no_slash == k_slash, (
        f"trailing slash produced different key: {k_no_slash!r} vs "
        f"{k_slash!r}. Path.resolve() is not doing its job.")
    assert k_no_slash == k_rel, (
        f"relative path resolved differently: {k_no_slash!r} vs "
        f"{k_rel!r}")


def test_legacy_key_file_still_hits_after_key_change(tmp_path):
    """Pod3d files written under the pre-resolve _key_raw formula must
    still HIT the cache after we switched to resolve(). This preserves
    the existing user's basis files during the key-formula transition
    so they don't have to re-fit."""
    sims = _mock_sims()
    # Manually write a file under the LEGACY hash: force _key_raw's
    # exact formula by passing a value that resolve() would change.
    real = tmp_path / "d"
    real.mkdir()
    legacy_path_str = str(real) + "/"          # resolve() strips this
    n_fit = len(sims)
    legacy_hash = _key_raw(legacy_path_str, 16, 16, 8, 1.0, 1.0, 0,
                            7, 0.8, 0.1, n_fit)
    new_hash = _key(legacy_path_str, 16, 16, 8, 1.0, 1.0, 0,
                     7, 0.8, 0.1, n_fit)
    assert legacy_hash != new_hash, (
        "test fixture wrong: legacy and resolved keys should differ "
        "for a trailing-slash npz_dir; if they match, resolve() did "
        "not strip the slash on this platform.")

    # Pre-populate a fake basis file under the legacy key.
    fit = PODBasis.fit(sims, K=8, verbose=False)
    legacy_file = tmp_path / f"pod3d_{legacy_hash}.npz"
    np.savez(legacy_file, Phi=fit.Phi, sigma=fit.sigma,
             spatial_shape=np.asarray(fit.spatial_shape,
                                       dtype=np.int64),
             k_cache=np.asarray(fit.Phi.shape[1], dtype=np.int64))
    assert not (tmp_path / f"pod3d_{new_hash}.npz").exists()

    # A load call with the same trailing-slash spelling should HIT the
    # legacy file (via the fallback) instead of re-fitting.
    basis = load_or_fit_basis(
        sims, K=4, npz_dir=legacy_path_str,
        nx=16, ny=16, nt=8, x_end=1.0, y_end=1.0,
        drop_first_steps=0, seed=7, train_frac=0.8, val_frac=0.1,
        cache_dir=tmp_path, k_cache=8)
    assert isinstance(basis, PODBasis)
    # The legacy file must still exist and no NEW file written for the
    # resolved key (fallback path just READ, did not create).
    assert legacy_file.exists()
    assert not (tmp_path / f"pod3d_{new_hash}.npz").exists(), (
        "legacy-fallback HIT should not have written a new pod3d "
        "under the resolved key")


def test_different_key_writes_separate_file(tmp_path):
    """Changing any keyed parameter (e.g. drop_first_steps) must
    produce a separate cache file -- two configurations must never
    share a basis fit."""
    sims = _mock_sims()
    kw_a = _kwargs(tmp_path)
    kw_b = dict(kw_a)
    kw_b["drop_first_steps"] = 1
    load_or_fit_basis(sims, K=4, **kw_a, k_cache=8)
    load_or_fit_basis(sims, K=4, **kw_b, k_cache=8)
    cached = sorted(tmp_path.glob("pod3d_*.npz"))
    assert len(cached) == 2, f"expected 2 distinct cache files, got {len(cached)}"
