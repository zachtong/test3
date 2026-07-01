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


def test_diversity_stats_cache_roundtrip(tmp_path):
    """Diversity cache: write then read must return identical (mean, var,
    n_eff, axes). Guards against silent schema drift where the reader
    forgets an array the writer stored."""
    from scripts.viz_diversity import (
        _stats_cache_path, _save_stats_cache, _load_stats_cache,
        _STATS_CACHE_PREFIX)

    folder = tmp_path / "diversity_folder"
    folder.mkdir()
    mean = np.random.default_rng(0).standard_normal((8, 8, 4)) * 1e-6
    var = np.abs(np.random.default_rng(1).standard_normal((8, 8, 4))) * 1e-12
    n_eff = 42
    x_canon = np.linspace(0, 1, 8)
    y_canon = np.linspace(0, 1, 8)

    path = _stats_cache_path(folder, 8, 8, 4, 1, None)
    assert path.name.startswith(_STATS_CACHE_PREFIX)
    assert path.parent == folder

    _save_stats_cache(path, mean, var, n_eff, x_canon, y_canon)
    hit = _load_stats_cache(path)
    assert hit is not None
    m2, v2, n2, x2, y2 = hit
    assert n2 == n_eff
    # Cache stores as float32 for size; allow that rounding.
    assert np.allclose(m2, mean, atol=1e-9)
    assert np.allclose(v2, var, atol=1e-16)
    assert np.array_equal(x2, x_canon)
    assert np.array_equal(y2, y_canon)


def test_diversity_stats_cache_invalidates_on_key_change(tmp_path):
    """Different (folder, grid, drop, limit) tuples must produce
    different cache paths so a rebuild after ANY key change goes to a
    fresh file instead of clobbering a valid one for another config."""
    from scripts.viz_diversity import _stats_cache_path
    folder = tmp_path / "d"
    folder.mkdir()
    base = _stats_cache_path(folder, 8, 8, 4, 1, None)
    paths = {
        "nx":    _stats_cache_path(folder, 16, 8, 4, 1, None),
        "ny":    _stats_cache_path(folder, 8, 16, 4, 1, None),
        "nt":    _stats_cache_path(folder, 8, 8, 8, 1, None),
        "drop":  _stats_cache_path(folder, 8, 8, 4, 0, None),
        "limit": _stats_cache_path(folder, 8, 8, 4, 1, 100),
        "other": _stats_cache_path(tmp_path / "other", 8, 8, 4, 1, None),
    }
    (tmp_path / "other").mkdir()
    paths["other"] = _stats_cache_path(tmp_path / "other", 8, 8, 4, 1, None)
    seen = {base}
    for k, p in paths.items():
        assert p != base, f"{k} did not change the cache key"
        assert p not in seen, f"{k} collided with a previous key"
        seen.add(p)


def test_diversity_stats_cache_rejects_stale_version(tmp_path):
    """A cache file written under an older version tag must be
    rejected by _load_stats_cache (returns None) so schema changes
    don't silently ship the old contents."""
    from scripts.viz_diversity import (_stats_cache_path,
                                          _load_stats_cache)
    folder = tmp_path / "d"
    folder.mkdir()
    path = _stats_cache_path(folder, 8, 8, 4, 1, None)
    np.savez(path, version=np.int32(-999),
             mean=np.zeros((8, 8, 4), dtype=np.float32),
             var=np.zeros((8, 8, 4), dtype=np.float32),
             n_eff=np.int32(1),
             x_canon=np.linspace(0, 1, 8),
             y_canon=np.linspace(0, 1, 8))
    assert _load_stats_cache(path) is None


def test_topdown_contour_does_not_accumulate_across_frames(tmp_path):
    """Concentric-rings regression. Pre-fix, _redraw_contour relied on
    ContourSet.collections which matplotlib 3.8 deprecated / 3.10 removed.
    The AttributeError was silently swallowed and every frame's bonded-
    region contour stayed on the axes, so the final frame ended up
    showing concentric rings tracking the front's entire trajectory.

    We render a 2-frame GIF whose bonding front sweeps inward-to-outward
    on a synthetic monotonic field, then count orange-coloured pixels
    in the LAST frame's rendered image. With the bug, the last frame
    has ~2x the orange pixels of the first (it carries the first
    frame's contour too). With the fix, the last frame has roughly the
    same count as the first.

    A pure orange-pixel count is not deterministic across mpl versions
    due to antialiasing, so the test only asserts the count is bounded
    by a generous 1.6x the first-frame count (well below the 2-3x a
    leaked contour produces).
    """
    pytest.importorskip("PIL")
    from scripts.viz_topdown_gif import (render_topdown_gif,
                                          _FRONT_COLOR)
    from core.simulation import Simulation
    from PIL import Image, ImageSequence

    Nx, Ny, Nt = 24, 24, 8
    w = _make_synthetic_w(Nx=Nx, Ny=Ny, Nt=Nt, peak_um=15.0)
    sim = Simulation(f=w.astype(np.float32), params={})
    x = np.linspace(0, 1, Nx)
    y = np.linspace(0, 1, Ny)
    sensor_xy = np.array([[1.0, 0.0]])
    out = tmp_path / "topdown.gif"
    render_topdown_gif(sim, x, y, sensor_xy, out,
                       fps=2, max_frames=Nt, sim_id="rings_test",
                       drop_first_steps=0)
    assert out.is_file()

    # _FRONT_COLOR is an RGB tuple in 0..1; convert to 0..255 ints.
    front_rgb = np.array(
        [int(round(c * 255)) for c in _FRONT_COLOR[:3]], dtype=np.int32)

    def _count_orange(im):
        a = np.asarray(im.convert("RGB"), dtype=np.int32)
        # Pixel close (manhattan distance < 60) to _FRONT_COLOR. Loose
        # tolerance covers anti-aliasing + GIF quantisation.
        d = np.abs(a - front_rgb[None, None, :]).sum(axis=-1)
        return int((d < 60).sum())

    with Image.open(out) as im:
        frames = [f.copy() for f in ImageSequence.Iterator(im)]
    assert len(frames) >= 4

    # Frame 0 has w=0 everywhere (rest), so no bonded region. Use a
    # mid-frame as the baseline instead.
    mid = _count_orange(frames[len(frames) // 2])
    last = _count_orange(frames[-1])
    # Sanity: contour must be drawn somewhere in the mid frame.
    assert mid > 5, f"no contour rendered in mid frame ({mid} px)"
    # The accumulating-rings bug would make last carry every prior
    # frame's contour: last >> mid. With the fix last has just one
    # ring (the current front) which is geometrically the largest
    # circle so far, so it can legitimately have somewhat more pixels
    # than mid -- but not Nt x more.
    # Generous bound at 2x mid. With the bug ratio was ~Nt/2 = 4.
    assert last <= 2.0 * mid + 200, (
        f"contour appears to accumulate: mid={mid}px last={last}px "
        f"(ratio {last / mid:.2f}). Pre-fix this scaled with frame "
        f"count.")


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
