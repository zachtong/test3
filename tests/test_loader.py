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

from tests.mock_data import make_mock_3d_npz                # noqa: E402
from data.loader import load_dataset                        # noqa: E402
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
    """A sim with a negative-x native coord must abort the load."""
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
    with pytest.raises(ValueError, match="first quadrant"):
        load_dataset(folder, nx=16, ny=16, nt=8, cache=False, workers=1)
