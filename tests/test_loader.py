"""Smoke tests: 3D loader against the documented NPZ schema.

These tests build fixtures matching docs/NPZ_SCHEMA.md (variable per-step
mesh sizes, quarter-disk coords, monotonic-with-boundary-duplicate tReal)
and verify the loader's two main contracts:

  1. `load_dataset` returns sims with `sim.f` shaped (Nx, Ny, Nt) on the
     canonical quarter-disk grid; sensor indexing and PODBasis.fit work
     on those sims.
  2. The cache round-trips: a second call returns identical fields and
     params without re-running the spatial interpolation pipeline.
"""

from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pytest

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from tests.mock_data import make_mock_3d_npz, make_bad_3d_npz   # noqa: E402
from data.loader import (load_dataset, preflight_npz,        # noqa: E402
                          _resolve_workers, _DEFAULT_WORKER_CAP)
from core.sensors import SensorConfig, place_sensors, sensor_indices  # noqa: E402
from core.pod_basis import PODBasis                         # noqa: E402
from core.grid import canonical_grid                        # noqa: E402


@pytest.fixture
def fixture_folder(tmp_path):
    """Two mock sims with slightly different shapes; common Nt downstream."""
    folder = tmp_path / "mock_3d_npz"
    folder.mkdir()
    make_mock_3d_npz(folder / "data_00000000.npz",
                     n_steps=2, t_per_step=(5, 4),
                     n_upper_per_step=(50, 60),
                     n_lower_per_step=(40, 55), seed=0)
    make_mock_3d_npz(folder / "data_00000001.npz",
                     n_steps=3, t_per_step=(3, 4, 5),
                     n_upper_per_step=(45, 55, 65),
                     n_lower_per_step=(40, 48, 58), seed=1)
    return folder


def test_loader_shape_and_mask(fixture_folder):
    nx, ny, nt = 32, 32, 16
    x, y, sims = load_dataset(fixture_folder, nx=nx, ny=ny, nt=nt,
                              cache=False, workers=1)
    assert len(sims) == 2
    for s in sims:
        assert s.f.shape == (nx, ny, nt)
        assert s.f.dtype == np.float32
        # Quarter-disk: off-disk corners must be exactly zero.
        X, Y = np.meshgrid(x, y, indexing="ij")
        off_disk = (X * X + Y * Y) > 1.0
        # Field on off-disk cells, for every time step, must be 0.
        assert np.all(s.f[off_disk, :] == 0.0)
        # Params carries provenance.
        assert "basename" in s.params
        assert "contactTime" in s.params  # from mock metadata
        assert s.params["bonding_front"].shape == (nt,)


def test_sensor_extract_and_pod(fixture_folder):
    nx, ny, nt = 32, 32, 16
    _, _, sims = load_dataset(fixture_folder, nx=nx, ny=ny, nt=nt,
                              cache=False, workers=1)
    x, y = canonical_grid(nx, ny)
    scfg = SensorConfig(n=3, strategy="custom",
                        positions=((1.0, 0.0), (1.0, 45.0), (1.0, 90.0)))
    xy = place_sensors(scfg)
    ij = sensor_indices(xy, x, y)
    # Sensor traces: (B, n, Nt)
    y_batch = np.stack([s.f[ij[:, 0], ij[:, 1], :] for s in sims], axis=0)
    assert y_batch.shape == (2, 3, nt)
    # POD round-trip on this tiny set.
    basis = PODBasis.fit(sims, K=4)
    assert basis.Phi.shape == (nx * ny, 4)
    a = basis.project_sim(sims[0])
    assert a.shape == (4, nt)
    f_rec = basis.reconstruct(a)
    assert f_rec.shape == (nx, ny, nt)


def test_cache_roundtrip(fixture_folder):
    nx, ny, nt = 32, 32, 16
    x1, y1, sims_a = load_dataset(fixture_folder, nx=nx, ny=ny, nt=nt,
                                  cache=True, workers=1)
    cache_path = fixture_folder / f"_loader_cache_{nx}x{ny}x{nt}.npz"
    assert cache_path.exists()
    x2, y2, sims_b = load_dataset(fixture_folder, nx=nx, ny=ny, nt=nt,
                                  cache=True, workers=1)
    assert np.allclose(x1, x2)
    assert np.allclose(y1, y2)
    for a, b in zip(sims_a, sims_b):
        assert np.allclose(a.f, b.f)
        assert a.params["basename"] == b.params["basename"]


def test_trim_last_step_metadata(fixture_folder):
    """Loader exposes the converter's trim-last-step bookkeeping in params."""
    _, _, sims = load_dataset(fixture_folder, nx=16, ny=16, nt=8,
                              cache=False, workers=1)
    for s in sims:
        assert s.params["last_step_removed"] is True
        # original > converted by exactly 1 in the mock; real converter same.
        assert (s.params["num_original_wafer_steps"]
                == s.params["num_wafer_steps"] + 1)
        assert "step_metadata_json" in s.params
        assert "skipped_steps_json" in s.params


def test_quarter_validation(tmp_path):
    """A single sim with negative-x native coords now becomes a skip-with-
    reason; load_dataset raises RuntimeError because zero sims load."""
    folder = tmp_path / "bad_quadrant"
    folder.mkdir()
    p = folder / "data_00000000.npz"
    make_mock_3d_npz(p, n_steps=1, t_per_step=(3,),
                     n_upper_per_step=(20,), n_lower_per_step=(20,))
    # Sabotage: load, mutate coordinates_upper[0, 0] to negative, save back.
    with np.load(p, allow_pickle=True) as z:
        payload = {k: z[k] for k in z.files}
    payload["step_0000_coordinates_upper"][0, 0] = -0.5
    np.savez(p, **payload)
    with pytest.raises(RuntimeError, match="passed preflight"):
        load_dataset(folder, nx=16, ny=16, nt=8, cache=False, workers=1)


def _good_mock_kwargs():
    """Standard 'good' fixture kwargs reused across skip-tolerance tests."""
    return dict(n_steps=2, t_per_step=(5, 4),
                n_upper_per_step=(40, 50), n_lower_per_step=(30, 40))


@pytest.mark.parametrize("mode", [
    "nonzero_skipped", "missing_step", "wrong_disp_shape",
    "bad_quadrant", "sample_shape_off", "last_step_kept",
    "treal_huge_back",
])
def test_preflight_rejects_each_sabotage(tmp_path, mode):
    """Every documented bad-mode must fail preflight with a non-empty reason.

    Drives `make_bad_3d_npz` through every sabotage mode and asserts
    `preflight_npz` returns (False, reason). This is the contract the
    skip-tolerant loader depends on: if preflight ever silently passed a
    broken file, that file would crash _build_one inside a worker and
    take down the whole cache build.
    """
    p = tmp_path / "bad.npz"
    make_bad_3d_npz(p, mode=mode, **_good_mock_kwargs())
    ok, reason = preflight_npz(p)
    assert ok is False, f"mode={mode} unexpectedly passed preflight"
    assert reason and isinstance(reason, str)


def test_preflight_accepts_clean(tmp_path):
    """The standard good fixture must pass preflight."""
    p = tmp_path / "good.npz"
    make_mock_3d_npz(p, **_good_mock_kwargs())
    ok, reason = preflight_npz(p)
    assert ok is True, f"good fixture rejected: {reason}"


def test_nearest_fill_recovers_axis_strip(tmp_path):
    """Native points missing the x=0 strip should NOT leave a column
    of zeros in the loaded canonical field.

    Builds a mock NPZ then surgically removes every native point with
    x < x_threshold (mimicking the real-data condition where COMSOL
    rarely samples exactly on the y-axis), reloads via the loader, and
    asserts that column ix=0 of the loaded field is mostly non-zero
    inside the disk -- which only the nearest-neighbour fill in
    _precompute_bary can deliver. Without the fix this column would
    be all zeros."""
    folder = tmp_path / "axis_gap"
    folder.mkdir()
    p = folder / "data.npz"
    make_mock_3d_npz(p, n_steps=2, t_per_step=(5, 4),
                     n_upper_per_step=(200, 240),
                     n_lower_per_step=(180, 220), seed=0)

    # Remove all native points with x < 0.015 m (= 10% of R) from
    # step 0's upper coords AND displacement, simulating a gap on
    # the y-axis side of the canonical hull.
    R = 0.15
    X_GAP_M = 0.015
    with np.load(p, allow_pickle=True) as z:
        payload = {k: z[k] for k in z.files}
    coords = payload["step_0000_coordinates_upper"]               # (3, N)
    disp = payload["step_0000_displacement_z_corrected_upper"]    # (T, N)
    thk = payload["step_0000_thickness_upper"]                    # (T, N)
    keep_mask = coords[0] >= X_GAP_M
    n_before, n_after = coords.shape[1], int(keep_mask.sum())
    assert n_after < n_before, "test setup: expected some points to be cut"
    payload["step_0000_coordinates_upper"] = coords[:, keep_mask]
    payload["step_0000_displacement_z_corrected_upper"] = disp[:, keep_mask]
    payload["step_0000_thickness_upper"] = thk[:, keep_mask]
    payload["step_0000_num_upper_points"] = np.int64(n_after)
    # also adjust the per-sample convenience array for the samples
    # belonging to step 0
    s_step = payload["sample_step_index"]
    s_nu = payload["sample_num_upper_points"]
    s_nu_arr = np.asarray(s_nu).copy()
    s_nu_arr[s_step == 0] = n_after
    payload["sample_num_upper_points"] = s_nu_arr
    np.savez(p, **payload)

    _, _, sims = load_dataset(folder, nx=32, ny=32, nt=8,
                              cache=False, workers=1,
                              drop_first_steps=0)
    assert len(sims) == 1
    f = sims[0].f                                    # (32, 32, 8)
    # Column ix=0 of canonical at the last time index, inside the disk.
    # The disk at column 0 spans y in [0, x_canon[0]^2 + y^2 <= 1] =>
    # all rows are in disk (x=0 -> any y in [0, 1] satisfies y^2 <= 1).
    col0_last = f[0, :, -1]
    # Pre-fix this would be all-zero. With nearest-fill it must have
    # at least a few non-zero cells.
    nnz = int((np.abs(col0_last) > 1e-12).sum())
    assert nnz >= f.shape[1] // 4, (
        f"x=0 column at t=-1 has only {nnz}/{f.shape[1]} non-zero "
        f"cells; nearest-neighbour fill not working")


def test_field_covers_full_quarter_disk(tmp_path):
    """Regression: native coord -> canonical grid normalisation must cover
    the WHOLE quarter-disk, not just the (0..R) corner.

    Before the loader divided native coords by R, the Delaunay triangulation
    sat in the [0, 0.15] x [0, 0.15] corner of the canonical [0, 1] grid;
    everything outside that 15% x 15% square was off-hull and got masked
    to 0. This test asserts that the field at canonical (~0.5, ~0.5)
    -- well inside the quarter-disk but well outside the old broken
    coverage region -- has non-trivial signal at the last time step.
    """
    folder = tmp_path / "coverage"
    folder.mkdir()
    make_mock_3d_npz(folder / "data.npz", n_steps=2, t_per_step=(5, 4),
                     n_upper_per_step=(120, 150),
                     n_lower_per_step=(100, 130), seed=42)
    _, _, sims = load_dataset(folder, nx=64, ny=64, nt=8,
                              cache=False, workers=1)
    assert len(sims) == 1
    f = sims[0].f  # (64, 64, 8)
    # Canonical (0.5, 0.5) is index (32, 32) on a 64-grid.
    last_t = f[32, 32, -1]
    assert np.isfinite(last_t)
    assert abs(last_t) > 1e-4, (
        f"field at canonical (0.5, 0.5) at last time = {last_t}; "
        f"expected non-trivial signal. Coord normalisation regressed?")
    # And the edge (1.0, 0.0) which sits on the quarter-disk boundary
    # also needs to be reachable -- this is where the rig sensor lives.
    edge = f[-1, 0, -1]
    assert np.isfinite(edge), f"edge cell is NaN: {edge}"


def test_drop_first_steps(tmp_path):
    """drop_first_steps=1 removes step_0000 samples; resulting sim is
    shorter and the cache filename carries the _drop1 suffix so the
    two configurations never collide.
    """
    folder = tmp_path / "drop"
    folder.mkdir()
    # 2-step mock: step 0 has 5 samples, step 1 has 4. total S = 9.
    make_mock_3d_npz(folder / "a.npz", n_steps=2, t_per_step=(5, 4),
                     n_upper_per_step=(50, 60),
                     n_lower_per_step=(40, 55), seed=0)
    # Baseline: drop_first_steps=0 keeps all 9 samples.
    _, _, sims0 = load_dataset(folder, nx=16, ny=16, nt=8, cache=False,
                               workers=1, drop_first_steps=0)
    assert sims0[0].params["num_samples"] == 9
    assert sims0[0].params["n_samples_dropped_from_first_steps"] == 0
    # Drop step 0: remaining 4 samples come from step 1.
    _, _, sims1 = load_dataset(folder, nx=16, ny=16, nt=8, cache=False,
                               workers=1, drop_first_steps=1)
    assert sims1[0].params["num_samples"] == 4
    assert sims1[0].params["num_samples_raw"] == 9
    assert sims1[0].params["n_samples_dropped_from_first_steps"] == 5
    assert sims1[0].params["drop_first_steps"] == 1
    # Cache filenames differ so the two configurations never collide.
    _, _, _ = load_dataset(folder, nx=16, ny=16, nt=8, cache=True,
                           workers=1, drop_first_steps=0)
    _, _, _ = load_dataset(folder, nx=16, ny=16, nt=8, cache=True,
                           workers=1, drop_first_steps=1)
    assert (folder / "_loader_cache_16x16x8.npz").exists()
    assert (folder / "_loader_cache_16x16x8_drop1.npz").exists()


def test_drop_first_steps_too_many_raises(tmp_path):
    """drop_first_steps >= num_wafer_steps removes every sample; the
    per-sim _build_one_safe wraps the ValueError into a skip reason,
    so the file becomes a skip and (when it's the only file) the
    load_dataset call ends in a zero-sims RuntimeError."""
    folder = tmp_path / "drop_all"
    folder.mkdir()
    make_mock_3d_npz(folder / "a.npz", n_steps=2, t_per_step=(5, 4),
                     n_upper_per_step=(50, 60),
                     n_lower_per_step=(40, 55), seed=0)
    with pytest.raises(RuntimeError, match="passed preflight"):
        load_dataset(folder, nx=16, ny=16, nt=8, cache=False,
                     workers=1, drop_first_steps=5)


def test_preflight_tolerates_small_treal_overlap(tmp_path):
    """A backward jump of a few sub-step samples must NOT fail preflight.

    Real waferData boundaries can overlap by several timesteps (the
    converter writes the last few samples of step i and the first sample
    of step i+1 at nearly the same physical time). With
    _TREAL_BACKWARD_FACTOR = 10 this should still pass; the old
    'one-timestep' policy used to reject it.
    """
    p = tmp_path / "small_overlap.npz"
    make_mock_3d_npz(p, **_good_mock_kwargs())
    with np.load(p, allow_pickle=True) as z:
        payload = {k: z[k] for k in z.files}
    treal = payload["sample_tReal"].copy()
    dt = np.diff(treal)
    typ_dt = float(np.median(dt[dt > 0]))
    # Move the last sample back by 3 * typ_dt (within the 10x window).
    treal[-1] = treal[-2] - 3.0 * typ_dt
    payload["sample_tReal"] = treal
    np.savez(p, **payload)
    ok, reason = preflight_npz(p)
    assert ok is True, f"small overlap unexpectedly rejected: {reason}"


def test_load_dataset_skips_bad_keeps_good(tmp_path):
    """Mixed folder: 2 good + 2 bad NPZs -> load_dataset returns 2 sims.

    Covers the core skip-tolerant contract: a bad NPZ alongside good ones
    must not abort the load. The good sims' params['basename'] are checked
    against the actual good filenames to make sure we didn't accidentally
    keep one of the bad ones.
    """
    folder = tmp_path / "mixed"
    folder.mkdir()
    make_mock_3d_npz(folder / "good_00.npz", **_good_mock_kwargs())
    make_mock_3d_npz(folder / "good_01.npz", **_good_mock_kwargs())
    make_bad_3d_npz(folder / "bad_00.npz", mode="nonzero_skipped",
                    **_good_mock_kwargs())
    make_bad_3d_npz(folder / "bad_01.npz", mode="missing_step",
                    **_good_mock_kwargs())
    x, y, sims = load_dataset(folder, nx=16, ny=16, nt=8, cache=False,
                              workers=1)
    assert len(sims) == 2
    names = sorted(s.params["basename"] for s in sims)
    assert names == ["good_00.npz", "good_01.npz"]


def test_load_dataset_zero_good_raises(tmp_path):
    """Folder full of bad NPZs -> RuntimeError mentioning preflight."""
    folder = tmp_path / "all_bad"
    folder.mkdir()
    make_bad_3d_npz(folder / "bad_00.npz", mode="nonzero_skipped",
                    **_good_mock_kwargs())
    make_bad_3d_npz(folder / "bad_01.npz", mode="last_step_kept",
                    **_good_mock_kwargs())
    with pytest.raises(RuntimeError, match="passed preflight"):
        load_dataset(folder, nx=16, ny=16, nt=8, cache=False, workers=1)


def test_resolve_workers_caps_on_huge_core_host(monkeypatch):
    """On a fat node (256 cores) the auto worker count must not blow up.

    Pre-fix the loader resolved workers=None to cpu_count - 2 = 254
    processes on a 256-core server, each numpy doing all-core BLAS ->
    server near-OOM. Cap is 32 by default; env override (numeric) wins.
    """
    import data.loader as L
    monkeypatch.setattr(L.os, "cpu_count", lambda: 256)
    # n_files large, no explicit request -> capped at _DEFAULT_WORKER_CAP.
    assert _resolve_workers(None, 9999) == _DEFAULT_WORKER_CAP
    # Env override raises the cap.
    monkeypatch.setenv("WAFER3D_LOADER_WORKERS_CAP", "64")
    assert _resolve_workers(None, 9999) == 64
    # Env override LOWERS the cap (shared box).
    monkeypatch.setenv("WAFER3D_LOADER_WORKERS_CAP", "4")
    assert _resolve_workers(None, 9999) == 4
    # Garbage env value -> falls back to default cap, not a crash.
    monkeypatch.setenv("WAFER3D_LOADER_WORKERS_CAP", "not_a_number")
    assert _resolve_workers(None, 9999) == _DEFAULT_WORKER_CAP
    monkeypatch.delenv("WAFER3D_LOADER_WORKERS_CAP", raising=False)
    # Explicit request is respected as-is (still capped by n_files only).
    assert _resolve_workers(200, 9999) == 200
    assert _resolve_workers(200, 5) == 5
    # n_files small -> resolved cap is the file count.
    assert _resolve_workers(None, 3) == 3
    # Small host -> auto wins over cap.
    monkeypatch.setattr(L.os, "cpu_count", lambda: 8)
    assert _resolve_workers(None, 9999) == 6   # 8 - 2


def test_cache_records_skip_log(tmp_path):
    """Cache round-trip must preserve both loaded sims and the skip log.

    The skip log governs the cache-hit decision: on a re-run the loader
    must reproduce the same skip outcome from cache without re-running
    preflight on the bad files. If the cache forgets the skipped files,
    the (loaded U skipped) == folder-files check would fail and the
    cache would (wastefully) get rebuilt every run.
    """
    folder = tmp_path / "mixed_cached"
    folder.mkdir()
    make_mock_3d_npz(folder / "good_00.npz", **_good_mock_kwargs())
    make_bad_3d_npz(folder / "bad_00.npz", mode="nonzero_skipped",
                    **_good_mock_kwargs())
    # First call: builds cache + writes skip log.
    x1, y1, sims_a = load_dataset(folder, nx=16, ny=16, nt=8,
                                  cache=True, workers=1)
    assert len(sims_a) == 1
    cache_path = folder / "_loader_cache_16x16x8.npz"
    assert cache_path.exists()
    with np.load(cache_path, allow_pickle=True) as z:
        assert "skipped_files" in z.files
        assert "skip_reasons" in z.files
        assert list(map(str, z["skipped_files"])) == ["bad_00.npz"]
        assert list(map(str, z["basenames"])) == ["good_00.npz"]
    # Second call: cache hit, returns identical sims, doesn't re-run build.
    x2, y2, sims_b = load_dataset(folder, nx=16, ny=16, nt=8,
                                  cache=True, workers=1)
    assert len(sims_b) == 1
    assert sims_a[0].params["basename"] == sims_b[0].params["basename"]
    assert np.allclose(sims_a[0].f, sims_b[0].f)
