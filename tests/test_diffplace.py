"""Tests for differentiable sensor placement: the parameterizations,
their feasible-band projection, the uniform-outer init, and the
full per-epoch history + redraw roundtrip."""

from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import scripts.train_differentiable_placement as DP        # noqa: E402


_PARAMS = ["cartesian", "polar-rad", "polar-deg"]


def test_uniform_outer_init_on_outer_ring_spread_in_angle():
    r, th = DP._init_positions("uniform-outer", 6, r_max=0.98)
    assert np.allclose(r, 0.98)
    assert np.allclose(th, np.linspace(0, 90, 6))          # 0,18,...,90


def test_abcdef_and_json_init():
    r, th = DP._init_positions("abcdef", 6)
    assert th.tolist() == [0, 45, 90, 0, 45, 90]
    r2, th2 = DP._init_positions("[[0.5,10],[0.6,80]]", 2)
    assert r2.tolist() == [0.5, 0.6] and th2.tolist() == [10, 80]


@pytest.mark.parametrize("param", _PARAMS)
def test_projection_keeps_positions_in_feasible_band(param):
    dev = torch.device("cpu")
    # start OUT of band on purpose: r too big/small, theta out of [0,90]
    r0 = np.array([1.5, -0.3, 0.5, 0.9])
    th0 = np.array([-30.0, 120.0, 45.0, 200.0])
    place = DP._Placement(r0, th0, param, r_min=0.2, r_max=0.98,
                          device=dev)
    place.project_()
    rt = place.rtheta()
    assert (rt[:, 0] >= 0.2 - 1e-4).all() and (rt[:, 0] <= 0.98 + 1e-4).all()
    assert (rt[:, 1] >= -1e-3).all() and (rt[:, 1] <= 90 + 1e-3).all()


@pytest.mark.parametrize("param", _PARAMS)
def test_xy_roundtrips_through_rtheta(param):
    r0 = np.array([0.52, 0.847, 0.7])
    th0 = np.array([0.0, 45.0, 90.0])
    place = DP._Placement(r0, th0, param, 0.2, 0.98, torch.device("cpu"))
    rt = place.rtheta()
    assert np.allclose(rt[:, 0], r0, atol=1e-4)
    assert np.allclose(rt[:, 1], th0, atol=1e-3)
    # xy() must agree with the polar coordinates it reports
    xy = place.xy().detach().numpy()
    exp = np.stack([r0 * np.cos(np.deg2rad(th0)),
                    r0 * np.sin(np.deg2rad(th0))], axis=1)
    assert np.allclose(xy, exp, atol=1e-4)


@pytest.fixture
def tiny_basis_traj(tmp_path):
    nx = ny = 24
    K = 6
    rng = np.random.default_rng(0)
    Phi, _ = np.linalg.qr(rng.standard_normal((nx * ny, K)))
    bp = tmp_path / "pod3d.npz"
    np.savez(bp, Phi=Phi, sigma=np.linspace(5, 1, K),
             spatial_shape=np.array([nx, ny]), k_cache=K)
    tp = tmp_path / "traj.npz"
    np.savez(tp, a_train_val=rng.standard_normal((40, K, 30)))
    return str(bp), str(tp), K


def test_run_records_full_history_and_stays_in_band(tiny_basis_traj):
    bp, tp, K = tiny_basis_traj
    res = DP.run(bp, traj_path=tp, K=K, n=6, init="uniform-outer",
                 param="cartesian", epochs=12, r_min=0.2, r_max=0.98,
                 val_frac=0.25)
    pf = res["pos_full"]
    assert pf.shape == (13, 6, 2)                           # init + 12 epochs
    assert res["move_hist"].shape == (12, 6)
    assert (pf[:, :, 0] >= 0.2 - 1e-4).all()
    assert (pf[:, :, 1] >= -1e-3).all() and (pf[:, :, 1] <= 90 + 1e-3).all()
    # on-axis sensors (theta 0 and 90) are symmetry stationary: they must
    # never leave their mirror axis under the band projection
    assert pf[:, 0, 1].max() <= 1e-3                        # sensor at 0 stays 0
    assert pf[:, -1, 1].min() >= 90 - 1e-3                  # sensor at 90 stays 90


def test_history_save_load_roundtrip(tiny_basis_traj, tmp_path):
    bp, tp, K = tiny_basis_traj
    res = DP.run(bp, traj_path=tp, K=K, n=6, init="abcdef",
                 param="polar-deg", epochs=10, val_frac=0.25)
    hpath = tmp_path / "hist.npz"
    DP._save_history(res, hpath)
    h = DP._load_history(hpath)
    assert h["param"] == "polar-deg"
    assert h["pos_full"].shape == res["pos_full"].shape
    assert np.allclose(h["pos_full"], res["pos_full"])
    assert len(h["pos_hist"]) == len(res["pos_hist"])
    # redraw path must accept the loaded dict without retraining
    DP._render(h, tmp_path / "fig.png")
    DP._render_location_anim(h, tmp_path / "anim.gif", fps=8,
                             max_frames=8)
    assert (tmp_path / "fig.png").stat().st_size > 0
    assert (tmp_path / "anim.gif").stat().st_size > 0
