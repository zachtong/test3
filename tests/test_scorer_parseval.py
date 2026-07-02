"""Regression: cached-norm scoring == full-F scoring (Parseval identity).

evaluate_from_cached_norms in evaluation/scorer.py uses:

    ||f_pred - f_true||^2 = ||a_pred - a_true||^2 + ||f_perp||^2
    ||f_true||^2           = ||a_true||^2 + ||f_perp||^2

with a_true = Phi^T @ f_true and f_perp = f_true - Phi @ a_true.
This identity requires Phi to have orthonormal columns.

Test builds a small synthetic setup, computes field_errs / floor_errs
BOTH ways, and asserts they agree to float32 precision. If this test
fails the trajectory cache produces wrong metrics and the training
loop lies to the operator about model quality.
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from core.simulation import Simulation                       # noqa: E402
from core.pod_basis import PODBasis                          # noqa: E402
from training.normalization import NormStats                  # noqa: E402
from training.config import config_from_yaml                  # noqa: E402
from evaluation.scorer import (evaluate_ensemble,             # noqa: E402
                                evaluate_from_cached_norms)
from training.traj_cache import compute_test_field_norms      # noqa: E402


class _IdentityModel(nn.Module):
    """Trivial model: returns a fixed a_pred regardless of input."""
    def __init__(self, a_pred: np.ndarray):
        super().__init__()
        self.a = torch.tensor(a_pred, dtype=torch.float32)
        self._dummy = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return self.a.to(x.device)


def _mock_config():
    """Minimal ExperimentConfig via loading a real yaml, then override
    what evaluate_ensemble reads (basically nothing at this level)."""
    import tempfile
    cfg_yaml = """\
data:
  npz_dir: "fake"
  nx: 16
  ny: 16
  nt: 8
  x_end: 1.0
  y_end: 1.0
  t_end: 1.0
  train_frac: 0.8
  seed: 7
  drop_first_steps: 0
pod:
  k: 4
sensors:
  n: 2
  strategy: custom
  positions:
    - [0.5, 0.0]
    - [0.5, 90.0]
model:
  arch: bitcn
  channels: 8
  kernel: 3
  dilations: [1]
  dropout: 0.0
  causal: false
train:
  epochs: 1
  batch_size: 2
  lr_init: 0.001
  lr_final: 0.0001
  lr_schedule: cosine
  val_frac: 0.1
"""
    with tempfile.NamedTemporaryFile("w", suffix=".yaml",
                                       delete=False) as fp:
        fp.write(cfg_yaml)
        p = fp.name
    return config_from_yaml(p)


def test_cached_norm_scorer_matches_full_field_scorer():
    """Parseval-derived scoring must give identical field_errs and
    floor_errs to the full ||Phi @ a - f|| scoring path."""
    rng = np.random.default_rng(0)
    Nx, Ny, Nt, K, n_test = 16, 16, 8, 4, 6

    # Build a synthetic sim ensemble whose K-mode POD basis is
    # well-defined (no degeneracies).
    test_sims = []
    for _ in range(n_test):
        f = (rng.standard_normal((Nx, Ny, Nt)).astype(np.float32)
             * 1e-5)
        test_sims.append(Simulation(f=f, params={"basename": "s"}))

    basis = PODBasis.fit(test_sims, K=K)
    # Project GT -> a_true (n_test, K, Nt)
    a_true = np.stack([basis.project_sim(s) for s in test_sims])
    # Fake a_pred slightly off from a_true
    a_pred = a_true + rng.standard_normal(a_true.shape) * 1e-6

    # Fake target_stats that are identity so invert_norm does nothing
    target_stats = NormStats(
        mean=np.zeros((K, Nt), dtype=np.float64),
        std=np.ones((K, Nt), dtype=np.float64))

    # Fake x_test tensor and one model that returns a_pred verbatim.
    x_test = torch.zeros(n_test, 2, Nt, dtype=torch.float32)
    model = _IdentityModel(a_pred)

    cfg = _mock_config()

    # Path A: full-field scorer (touches test_sims.f)
    result_full = evaluate_ensemble(
        [model], x_test, test_sims, basis, target_stats, cfg,
        test_regimes=None)

    # Path B: cached-norm scorer (does NOT touch test_sims.f)
    f_true_sq, f_perp_sq = compute_test_field_norms(test_sims, a_true)
    result_cached = evaluate_from_cached_norms(
        [model], x_test, a_true, f_true_sq, f_perp_sq,
        [s.params["basename"] for s in test_sims],
        basis.sigma, target_stats, cfg)

    fa = np.asarray(result_full.per_sim_field_errs)
    fc = np.asarray(result_cached.per_sim_field_errs)
    la = np.asarray(result_full.per_sim_floor_errs)
    lc = np.asarray(result_cached.per_sim_floor_errs)

    # float32 accumulation in the full-field path bounds precision;
    # relative error on the order of 1e-4 is expected.
    assert np.allclose(fa, fc, rtol=5e-4, atol=1e-8), (
        f"field_errs disagree:\n  full={fa}\n  cached={fc}")
    assert np.allclose(la, lc, rtol=5e-4, atol=1e-8), (
        f"floor_errs disagree:\n  full={la}\n  cached={lc}")
    # gap must agree since both are ratios of the above.
    assert abs(result_full.gap_to_floor
                - result_cached.gap_to_floor) < 5e-3
