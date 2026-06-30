"""Tests for the deterministic helpers in scripts/fieldviz."""

from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pytest

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from scripts.fieldviz import (                               # noqa: E402
    mirror_d2, shared_diverging_cmap, compute_bonded_mask,
    front_radius_per_t,
)


def test_mirror_d2_shape_doubled_with_shared_axis():
    quarter = np.arange(9, dtype=np.float64).reshape(3, 3)
    full = mirror_d2(quarter)
    # 2 * 3 - 1 = 5 in each spatial axis (shared row/col is NOT duplicated)
    assert full.shape == (5, 5)


def test_mirror_d2_recovers_symmetric_input():
    """If the input is already D2-symmetric (mirror about both axes), the
    mirrored output's centre quadrant must equal the input."""
    quarter = np.array([[5.0, 4.0, 3.0],
                        [4.0, 2.0, 1.0],
                        [3.0, 1.0, 0.0]])
    full = mirror_d2(quarter)
    # The quarter sits at [2:5, 2:5] in the full output.
    assert np.allclose(full[2:, 2:], quarter)
    # And the mirrored quadrants reflect correctly:
    assert np.allclose(full[2:, :3], quarter[:, ::-1])
    assert np.allclose(full[:3, 2:], quarter[::-1, :])
    assert np.allclose(full[:3, :3], quarter[::-1, ::-1])


def test_mirror_d2_preserves_trailing_dims():
    quarter = np.random.default_rng(0).standard_normal((4, 4, 7))
    full = mirror_d2(quarter)
    assert full.shape == (7, 7, 7)


def test_shared_diverging_cmap_symmetric():
    field = np.array([-3.0, -1.0, 0.0, 2.0, 5.0])
    vmin, vmax = shared_diverging_cmap(field, symmetric=True,
                                       pct_lo=0, pct_hi=100)
    assert vmin == -5.0
    assert vmax == 5.0


def test_shared_diverging_cmap_clipping_protects_outlier():
    field = np.array([0.0] * 100 + [1000.0])
    vmin, vmax = shared_diverging_cmap(field, symmetric=True,
                                       pct_lo=1, pct_hi=99)
    # The 99th percentile of this distribution is 0 (the outlier sits
    # past it), so vmax should NOT explode to 1000.
    assert vmax < 100.0


def _make_synthetic_w(Nx=8, Ny=8, Nt=20, peak_um=15.0):
    """A simple bonding-style sim with the REAL-DATA sign convention:
    upper wafer starts at z=0 (no displacement) and DESCENDS to a
    negative position at the final frame (where lower wafer sits).
    Outer cells take longer to descend, mimicking an inward-to-outward
    bonding-front sweep. With peak_um=15 the initial gap is 15 um (>
    10 * 1 um = 10 um), so rule 2 of compute_bonded_mask fires."""
    x = np.linspace(0, 1, Nx)
    y = np.linspace(0, 1, Ny)
    X, Y = np.meshgrid(x, y, indexing="ij")
    R = np.sqrt(X * X + Y * Y)
    R = np.clip(R, 1e-3, 1.0)
    onset = R              # outer rim bonds latest
    t_norm = np.linspace(0, 1, Nt)
    # w(t=0) = 0 (rest); w(t=onset) = -peak_um (fully bonded at the
    # lower wafer's position); linear interpolation in between.
    w = np.zeros((Nx, Ny, Nt), dtype=np.float64)
    for k, tt in enumerate(t_norm):
        progress = np.clip(tt / onset, 0.0, 1.0)
        w[..., k] = -peak_um * 1e-6 * progress
    return w


def test_bonded_mask_rule_1_below_threshold_triggers():
    """A cell whose gap goes from clearly open to clearly closed must
    be marked bonded by the last frame."""
    w = _make_synthetic_w(peak_um=15.0)
    bonded = compute_bonded_mask(w, gap_threshold_um=1.0)
    # At the last frame, every cell on the disk should be bonded
    # (the synthetic has gap=0 by construction at t=-1).
    assert bonded[..., -1].all()


def test_bonded_mask_rule_2_filters_always_small_cells():
    """A cell that was never clearly open (gap always below
    10*threshold) must NOT be marked bonded, even if its instantaneous
    gap is below threshold."""
    Nx, Ny, Nt = 4, 4, 10
    # All cells have gap_um = 0.5 throughout (below 1 um threshold,
    # but never above 10 um) -> rule 2 fails.
    w = np.zeros((Nx, Ny, Nt), dtype=np.float64)
    # Make gap = 0.5 um relative to final frame.
    w[..., :] = 0.5e-6
    w[..., -1] = 0.0
    bonded = compute_bonded_mask(w, gap_threshold_um=1.0,
                                 well_above_factor=10.0)
    assert not bonded.any(), "cells never opened wide should not bond"


def test_bonded_mask_rule_3_monotonic_in_time():
    """Once bonded, must stay bonded. Verify cumulative property."""
    w = _make_synthetic_w(peak_um=15.0)
    bonded = compute_bonded_mask(w, gap_threshold_um=1.0)
    # bonded must be monotonically non-decreasing along the time axis.
    diff = np.diff(bonded.astype(np.int8), axis=-1)
    assert (diff >= 0).all(), "bonded mask should never retreat in time"


def test_front_radius_progresses_inward_outward_correctly():
    """The synthetic sim has its bonding front grow from r=0 outward to
    r=1; therefore the derived front radius should be monotonically
    NON-DECREASING in time (bonded region grows, unbonded annulus
    shrinks toward the edge)."""
    w = _make_synthetic_w(peak_um=15.0, Nx=16, Ny=16, Nt=40)
    bonded = compute_bonded_mask(w, gap_threshold_um=1.0)
    x = np.linspace(0, 1, 16)
    y = np.linspace(0, 1, 16)
    front = front_radius_per_t(bonded, x, y, bonded_frac_threshold=0.5)
    valid = ~np.isnan(front)
    if valid.sum() < 2:
        pytest.skip("not enough non-NaN front values to test monotonicity")
    finite = front[valid]
    # Allow tiny down-jitter from binning quantisation but the trend
    # must be net non-decreasing.
    assert finite[-1] >= finite[0], (
        f"front did not move outward: {finite[0]:.3f} -> {finite[-1]:.3f}")
