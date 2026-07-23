"""Tests for the tool-CSV ingest: marker parsing, nm->m / ms->s conversion,
zero subtraction, and missing-sensor handling."""

from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pytest

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from data.csv_ingest import csv_to_raw_dict, read_csv           # noqa: E402

_CSV = """\
Detect time[ms]
,XM,XE,YM,YE,DM,DE
,5,5,5,5,5,5
Zero position[nm]
,XM,XE,YM,YE,DM,DE
,10,20,10,20,10,20
Sampling Data[nm]
Time[ms],XM,XE,YM,YE,DM,DE
0,-100,-200,-110,-210,-105,-205
1000,-150,-250,-160,-260,-155,-255
2000,-200,-300,-210,-310,-205,-305
"""


def _write(tmp_path, text=_CSV):
    p = tmp_path / "run.csv"
    p.write_text(text)
    return p


def test_csv_to_raw_dict_units_and_zero(tmp_path):
    raw = csv_to_raw_dict(_write(tmp_path))
    assert np.allclose(raw["time"], [0.0, 1.0, 2.0])          # ms -> s
    # w = (sampling_nm - zero_nm) * 1e-9, in metres
    assert np.allclose(raw["w_XM"], np.array([-110, -160, -210]) * 1e-9)
    assert np.allclose(raw["w_XE"], np.array([-220, -270, -320]) * 1e-9)
    assert set(raw) == {"time", "w_XM", "w_XE", "w_YM", "w_YE", "w_DM", "w_DE"}


def test_missing_sensor_column_is_dropped(tmp_path):
    # a sensor whose column is all "--" must not appear in the raw dict
    text = _CSV.replace(
        "Time[ms],XM,XE,YM,YE,DM,DE",
        "Time[ms],XM,XE,YM,YE,DM,DE,ZZ").replace(
        "0,-100,-200,-110,-210,-105,-205",
        "0,-100,-200,-110,-210,-105,-205,--").replace(
        "1000,-150,-250,-160,-260,-155,-255",
        "1000,-150,-250,-160,-260,-155,-255,--").replace(
        "2000,-200,-300,-210,-310,-205,-305",
        "2000,-200,-300,-210,-310,-205,-305,--")
    raw = csv_to_raw_dict(_write(tmp_path, text))
    assert "w_ZZ" not in raw
    assert "w_XM" in raw


def test_nan_zero_treated_as_zero(tmp_path):
    text = _CSV.replace(",10,20,10,20,10,20", ",--,20,10,20,10,20")
    raw = csv_to_raw_dict(_write(tmp_path, text))
    # XM zero is now missing -> treated as 0, so w = sampling * 1e-9
    assert np.allclose(raw["w_XM"], np.array([-100, -150, -200]) * 1e-9)


def test_missing_data_block_raises(tmp_path):
    p = tmp_path / "bad.csv"
    p.write_text("Zero position[nm]\n,XM\n,10\n")
    with pytest.raises(ValueError, match="Sampling Data"):
        read_csv(p)
