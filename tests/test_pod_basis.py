"""Tests for the streaming-Gram PODBasis.fit + parallel reduce."""

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


def _mock_sims(n_sim: int = 8, nx: int = 16, ny: int = 16,
               nt: int = 6, seed: int = 0):
    rng = np.random.default_rng(seed)
    return [Simulation(f=rng.standard_normal((nx, ny, nt)).astype(np.float32),
                       params={})
            for _ in range(n_sim)]


def test_serial_fit_recovers_basic_pod():
    sims = _mock_sims()
    K = 4
    basis = PODBasis.fit(sims, K=K, verbose=False)
    assert basis.Phi.shape == (16 * 16, K)
    assert basis.sigma.shape == (K,)
    # sigma should be monotonically non-increasing.
    assert np.all(np.diff(basis.sigma) <= 0)
    # Phi columns should be orthonormal.
    gram = basis.Phi.T @ basis.Phi
    assert np.allclose(gram, np.eye(K), atol=1e-8)


def test_parallel_fit_matches_serial(tmp_path):
    """Parallel partial-Gram reduce must produce the same singular
    spectrum as serial. Modes can flip sign (eigenvector ambiguity)
    so we compare the absolute Phi columns and the sigma values
    directly.
    """
    sims = _mock_sims(n_sim=12)
    K = 4
    serial = PODBasis.fit(sims, K=K, verbose=False, workers=None)
    parallel = PODBasis.fit(sims, K=K, verbose=False, workers=3)
    assert np.allclose(serial.sigma, parallel.sigma, rtol=1e-10)
    # Sign-invariant column comparison: abs(<phi_serial, phi_parallel>) ~ 1
    for k in range(K):
        ip = abs(float(serial.Phi[:, k] @ parallel.Phi[:, k]))
        assert ip > 1 - 1e-8, f"mode {k} mismatch: |<.,.>| = {ip}"


def test_parallel_fit_handles_more_workers_than_sims():
    """If workers > n_sims the fit should fall back to serial-like
    behaviour rather than spawn empty workers."""
    sims = _mock_sims(n_sim=3)
    basis = PODBasis.fit(sims, K=2, verbose=False, workers=8)
    assert basis.Phi.shape == (16 * 16, 2)
    assert basis.sigma.shape == (2,)


def test_reconstruct_round_trip():
    """Project then reconstruct: with K = n_snap the reconstruction
    must recover the field to numerical precision."""
    sims = _mock_sims(n_sim=2, nt=4)
    n_snap = 2 * 4
    basis = PODBasis.fit(sims, K=n_snap, verbose=False)
    a = basis.project_sim(sims[0])
    f_rec = basis.reconstruct(a)
    assert np.allclose(f_rec, sims[0].f, atol=1e-5)
