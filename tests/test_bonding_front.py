"""Tests for the bonding-front detection ported from the 2D GUI: the
cumulative bonded mask (threshold + hysteresis + never-retreat) and the
per-azimuth front radius on the full (x, y) grid."""

from __future__ import annotations
import sys
from pathlib import Path

import numpy as np

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from evaluation.bonding_front import (bonded_mask, front_radii,   # noqa: E402
                                      sample_nearest)


def test_bonded_mask_needs_hysteresis_and_never_retreats():
    # a cell that starts AT its final value (never clearly open) must NOT bond,
    # even though its gap is always below threshold.
    w_flat = np.full((1, 1, 6), -5.0e-6)
    assert not bonded_mask(w_flat).any()
    # a cell that opens (gap >> threshold) then closes to final -> bonds and
    # STAYS bonded for every later frame (cumulative, no retreat).
    w = np.array([-1.0e-6, -2.0e-6, -3.0e-6, -5.0e-6, -5.0e-6, -5.0e-6])
    m = bonded_mask(w.reshape(1, 1, 6))[0, 0]
    assert not m[0] and m[-1]
    assert np.all(np.maximum.accumulate(m) == m)     # monotone in time


def test_front_radius_matches_a_known_bonded_disk():
    # build a full-disk mask: bonded inside r<=0.4, unbonded outside, at frame j.
    n = 81
    xf = np.linspace(-1.0, 1.0, n)
    yf = np.linspace(-1.0, 1.0, n)
    X, Y = np.meshgrid(xf, yf, indexing="ij")
    inside = (X * X + Y * Y) <= 0.4 ** 2
    thetas = np.linspace(0, 2 * np.pi, 60, endpoint=False)
    r = front_radii(inside.astype(float), xf, yf, thetas)
    r = r[np.isfinite(r)]
    assert r.size > 0
    assert np.allclose(r, 0.4, atol=0.05)            # ring sits at r ~ 0.4


def test_front_is_nan_when_fully_open_or_fully_bonded():
    n = 41
    xf = yf = np.linspace(-1.0, 1.0, n)
    thetas = np.linspace(0, 2 * np.pi, 24, endpoint=False)
    X, Y = np.meshgrid(xf, yf, indexing="ij")
    disk = (X * X + Y * Y) <= 1.0
    open_mask = np.zeros((n, n))                      # nothing bonded
    full_mask = disk.astype(float)                    # everything bonded
    assert np.all(np.isnan(front_radii(open_mask, xf, yf, thetas)))
    assert np.all(np.isnan(front_radii(full_mask, xf, yf, thetas)))


def test_sample_nearest_passes_nan_through():
    xf = yf = np.linspace(-1.0, 1.0, 5)
    field = np.arange(25.0).reshape(5, 5)
    px = np.array([0.0, np.nan])
    py = np.array([0.0, 0.0])
    out = sample_nearest(field, xf, yf, px, py)
    assert np.isfinite(out[0]) and np.isnan(out[1])
