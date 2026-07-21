"""Tests for the multi-start search helpers: Latin-hypercube sampling,
structured seeds, and permutation-invariant dedup of layouts."""

from __future__ import annotations
import sys
from pathlib import Path

import numpy as np

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import run_diffplace_multistart as MS                        # noqa: E402


def test_lhs_is_stratified_per_axis():
    rng = np.random.default_rng(0)
    N, d = 20, 12
    U = MS._lhs(N, d, rng)
    assert U.shape == (N, d)
    assert U.min() >= 0.0 and U.max() <= 1.0
    for j in range(d):
        bins = np.floor(U[:, j] * N).astype(int)
        assert len(set(bins.tolist())) == N     # one sample per stratum


def test_lhs_layouts_shape_and_bounds():
    rng = np.random.default_rng(1)
    lays = MS._lhs_layouts(7, 6, 0.2, 0.98, rng)
    assert len(lays) == 7
    for p in lays:
        assert p.shape == (6, 2)
        assert (p[:, 0] >= 0.2 - 1e-9).all() and (p[:, 0] <= 0.98 + 1e-9).all()
        assert (p[:, 1] >= 0.0).all() and (p[:, 1] <= 90.0).all()


def test_structured_seeds_are_valid_layouts():
    s = MS._structured_seeds(6, 0.2, 0.98)
    assert len(s) >= 3
    for p in s:
        assert p.shape == (6, 2)
        assert (p[:, 1] >= 0.0).all() and (p[:, 1] <= 90.0).all()


def test_same_layout_is_permutation_invariant():
    A = np.array([[0.5, 10], [0.8, 40], [0.3, 70],
                  [0.9, 20], [0.4, 55], [0.7, 85]])
    A_perm = A[[3, 0, 5, 1, 4, 2]]                # same set, relabeled
    assert MS._same_layout(A, A_perm, 0.06, 10.0) is True


def test_same_layout_rejects_a_moved_sensor():
    A = np.array([[0.5, 10], [0.8, 40], [0.3, 70],
                  [0.9, 20], [0.4, 55], [0.7, 85]])
    B = A.copy()
    B[0, 0] += 0.5                                # move one sensor a lot
    assert MS._same_layout(A, B, 0.06, 10.0) is False


def test_dedup_merges_permutations_keeps_distinct():
    A = np.array([[0.5, 10], [0.8, 40], [0.3, 70],
                  [0.9, 20], [0.4, 55], [0.7, 85]])
    A_perm = A[[3, 0, 5, 1, 4, 2]]
    B = A.copy()
    B[0, 0] += 0.5
    runs = [dict(best_pos=A, best_val=1.0),
            dict(best_pos=A_perm, best_val=1.1),
            dict(best_pos=B, best_val=1.2)]
    kept = MS._dedup_order(runs, [0, 1, 2], 0.06, 10.0)
    assert kept == [0, 2]                         # perm of #0 merged out
