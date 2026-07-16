"""Tests for the 3D real-experiment adapter: azimuth fold + per-(r, theta)
channel matching with NO averaging (the 2D->3D difference)."""

from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pytest

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from data.real_experiment import (                            # noqa: E402
    Channel, RealExperimentConfig, assemble_inputs, fold_theta)


# --- the ABCDEF layout the six real sensors fold onto ---
_ABCDEF = np.array([[0.52, 0.0], [0.52, 45.0], [0.52, 90.0],
                    [0.847, 0.0], [0.847, 45.0], [0.847, 90.0]])
_KEYS = ["w_XM", "w_DM", "w_YM", "w_XE", "w_DE", "w_YE"]


def _six_channels():
    """The six real sensors with PHYSICAL azimuths (X=180, Y=90, D=-45),
    which fold onto theta {0, 45, 90}."""
    return (
        Channel("w_XM", 0.078, fold_theta(180)),
        Channel("w_DM", 0.078, fold_theta(-45)),
        Channel("w_YM", 0.078, fold_theta(90)),
        Channel("w_XE", 0.127, fold_theta(180)),
        Channel("w_DE", 0.127, fold_theta(-45)),
        Channel("w_YE", 0.127, fold_theta(90)),
    )


def _raw(T=50, t0=0.0, t1=13.0):
    t = np.linspace(t0, t1, T)
    raw = {"time": t}
    for i, k in enumerate(_KEYS):
        raw[k] = (i + 1) * np.sin(np.linspace(0, 3, T))       # distinct per key
    return raw


def _cfg(**kw):
    base = dict(R=0.15, channels=_six_channels(), t_cutoff=13.0, t_start=0.0)
    base.update(kw)
    return RealExperimentConfig(**base)


@pytest.mark.parametrize("physical,folded", [
    (0, 0.0), (45, 45.0), (90, 90.0), (180, 0.0), (-45, 45.0),
    (270, 90.0), (225, 45.0), (135, 45.0), (360, 0.0),
])
def test_fold_theta(physical, folded):
    assert fold_theta(physical) == folded


def test_folded_theta_lands_in_quarter():
    for deg in range(-360, 361, 7):
        assert 0.0 <= fold_theta(deg) <= 90.0


def test_rows_map_by_position_and_are_not_averaged():
    raw = _raw()
    y, t = assemble_inputs(raw, _ABCDEF, _cfg())
    assert y.shape[0] == 6
    # each row must equal exactly ONE key's series (no azimuthal averaging)
    for i, k in enumerate(_KEYS):
        assert np.allclose(y[i], raw[k]), f"row {i} is not the raw {k} series"


def test_row_order_follows_sensor_rtheta_not_config_order():
    raw = _raw()
    # request a permuted order; rows must follow the requested positions
    order = [5, 0, 3, 2, 1, 4]
    perm = _ABCDEF[order]
    y, _ = assemble_inputs(raw, perm, _cfg())
    for out_row, src in enumerate(order):
        assert np.allclose(y[out_row], raw[_KEYS[src]])


def test_rounded_bundle_radius_still_matches():
    # bundle stores 0.847; the channel is 0.127/0.15 = 0.84667 -- within r_tol
    raw = _raw()
    y, _ = assemble_inputs(raw, [[0.847, 0.0]], _cfg())
    assert np.allclose(y[0], raw["w_XE"])


def test_window_truncates_and_dedups():
    raw = _raw(T=40, t0=0.0, t1=20.0)
    # inject a duplicate timestamp inside the window
    raw["time"][5] = raw["time"][4]
    y, t = assemble_inputs(raw, _ABCDEF, _cfg(t_start=2.0, t_cutoff=10.0))
    assert t.min() >= 2.0 and t.max() <= 10.0
    assert np.all(np.diff(t) > 0)                              # strictly increasing


def test_sign_flip_and_zero_baseline():
    raw = _raw()
    y_plain, _ = assemble_inputs(raw, [[0.52, 0.0]], _cfg())
    y_neg, _ = assemble_inputs(raw, [[0.52, 0.0]], _cfg(sign=-1.0))
    assert np.allclose(y_neg[0], -y_plain[0])
    y_zb, _ = assemble_inputs(raw, [[0.52, 0.0]], _cfg(zero_baseline=True))
    assert y_zb[0, 0] == pytest.approx(0.0)


def test_ambiguous_position_raises():
    raw = _raw()
    bad = RealExperimentConfig(
        R=0.15, t_cutoff=13.0,
        channels=(Channel("w_XM", 0.078, 0.0), Channel("w_YM", 0.078, 0.0)))
    with pytest.raises(ValueError, match="ambiguous"):
        assemble_inputs(raw, [[0.52, 0.0]], bad)


def test_same_radius_different_theta_are_distinct_channels():
    # the whole point of 3D: (0.52, 0) and (0.52, 90) must NOT collide
    raw = _raw()
    y, _ = assemble_inputs(raw, [[0.52, 0.0], [0.52, 90.0]], _cfg())
    assert np.allclose(y[0], raw["w_XM"])
    assert np.allclose(y[1], raw["w_YM"])
    assert not np.allclose(y[0], y[1])


def test_missing_channel_raises():
    raw = _raw()
    del raw["w_YE"]
    with pytest.raises(KeyError, match="w_YE"):
        assemble_inputs(raw, _ABCDEF, _cfg())


def test_backward_time_raises():
    raw = _raw()
    raw["time"][10] = raw["time"][0] - 5.0                     # big backward jump
    with pytest.raises(ValueError, match="backward"):
        assemble_inputs(raw, _ABCDEF, _cfg())


def test_window_bounds_validated():
    raw = _raw(t1=13.0)
    with pytest.raises(ValueError, match="t_start < t_cutoff"):
        assemble_inputs(raw, _ABCDEF, _cfg(t_start=5.0, t_cutoff=3.0))
