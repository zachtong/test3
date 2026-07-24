"""Tests for real-data leave-one-out: identify the held-out sensor, and the
end-to-end predict-at-held-out vs measured comparison."""

from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import torch

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from core.grid import canonical_grid                          # noqa: E402
from core.sensors import (SensorConfig, place_sensors,        # noqa: E402
                          sensor_indices)
from models import create_model                               # noqa: E402
from data.real_experiment import real_config_from_yaml        # noqa: E402
import scripts.eval_real_loo as LOO                           # noqa: E402
from scripts.eval_real import _load_raw                       # noqa: E402

_A = (0.52, 0.0)
_B = (0.52, 45.0)
_C = (0.52, 90.0)
_D = (0.847, 0.0)
_E = (0.847, 45.0)
_F = (0.847, 90.0)


def test_leftout_finds_the_single_missing_sensor():
    lo = LOO._leftout([_A, _B, _C, _D, _E])                   # missing F
    assert [x[2] for x in lo] == ["F"]
    lo2 = LOO._leftout([_A, _B, _C, _E, _F])                  # missing D
    assert [x[2] for x in lo2] == ["D"]


def test_leftout_empty_for_full_and_finds_two_for_n4():
    assert LOO._leftout([_A, _B, _C, _D, _E, _F]) == []       # nothing held out
    lo = LOO._leftout([_A, _B, _C, _D])                       # missing E, F
    assert {x[2] for x in lo} == {"E", "F"}


def test_rel_l2():
    a = np.array([1.0, 2.0, 3.0])
    assert LOO._rel_l2(a, a) == 0.0
    assert LOO._rel_l2(np.zeros(3), a) == 1.0                 # ||0-a||/||a|| = 1


def _mk_bundle(positions, path, nx=32, ny=32, K=6, nt=20):
    x, y = canonical_grid(nx, ny)
    rng = np.random.default_rng(0)
    Phi, _ = np.linalg.qr(rng.standard_normal((nx * ny, K)))
    n = len(positions)
    scfg = SensorConfig(n=n, strategy="custom", positions=tuple(positions))
    sxy = place_sensors(scfg)
    m = create_model("bitcn", n_in=n, n_out=K, channels=8, dilations=(1, 2),
                     kernel=3, dropout=0.0, causal=False)
    torch.save(dict(
        tag=path.stem, uses_front=False, Phi=Phi, sigma=np.linspace(5, 1, K),
        spatial_shape=np.array([nx, ny]), x_canon=x, y_canon=y, nx=nx, ny=ny,
        nt=nt, K=K, x_end=1.0, y_end=1.0, sensor_rtheta=np.array(positions),
        sensor_xy=sxy, sensor_ij=sensor_indices(sxy, x, y),
        y_mean=np.zeros((1, n, 1)), y_std=np.ones((1, n, 1)),
        target_mean=np.zeros((1, K, 1)), target_std=np.ones((1, K, 1)),
        model=dict(arch="bitcn", n_in=n, n_out=K, channels=8,
                   dilations=[1, 2], kernel=3, dropout=0.0, causal=False),
        seeds=[7], state_dicts=[m.state_dict()]), path)


def test_grid_generation():
    assert LOO._grid(None, 7.5) == [7.5]                       # fixed axis
    assert LOO._grid([6.0, 8.0, 0.5], 0) == [6.0, 6.5, 7.0, 7.5, 8.0]


def _make_real(tmp_path, T=50):
    raw = {"time": np.linspace(0, 13, T)}
    for i, k in enumerate(["w_XM", "w_DM", "w_YM", "w_XE", "w_DE", "w_YE"]):
        raw[k] = -(i + 1) * 1e-6 * (1 - np.exp(-np.linspace(0, 13, T) / 3))
    np.savez(tmp_path / "real.npz", **raw)
    return tmp_path / "real.npz"


def test_loo_end_to_end(tmp_path):
    _mk_bundle([_A, _B, _C, _D, _E], tmp_path / "n5_ABCDE.pt")   # out F
    real = _make_real(tmp_path)
    cfg = real_config_from_yaml(str(_root / "configs" / "real_exp_n6.yaml"))
    r2 = _load_raw(str(real))
    L = LOO._load(str(tmp_path / "n5_ABCDE.pt"))
    recs = LOO._one_bundle(L, r2, cfg)
    assert len(recs) == 1
    rec = recs[0]
    assert rec["label"] == "F"                                # held-out sensor
    assert rec["pred"].shape == (20,) and rec["meas"].shape == (20,)
    assert np.isfinite(rec["rel_l2"])


def test_window_sweep_picks_best(tmp_path):
    import subprocess
    _mk_bundle([_A, _B, _C, _D, _E], tmp_path / "n5_ABCDE.pt")
    _mk_bundle([_A, _B, _C, _D, _F], tmp_path / "n5_ABCDF.pt")   # out E
    real = _make_real(tmp_path)
    out = tmp_path / "loo"
    r = subprocess.run(
        [sys.executable, "scripts/eval_real_loo.py",
         "--bundles", str(tmp_path / "n5_ABCDE.pt"),
         str(tmp_path / "n5_ABCDF.pt"), "--real", str(real),
         "--config", str(_root / "configs" / "real_exp_n6.yaml"),
         "--sweep-t-cutoff", "6", "12", "2", "--out-dir", str(out)],
        cwd=str(_root), capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-800:]
    run = out / "real"                                        # <out>/<csv stem>/
    assert (run / "loo_sweep.png").is_file()
    assert (run / "loo.png").is_file()
    # one per-model figure per bundle, held-out sensor hollow
    assert (run / "model_n5_ABCDE.png").is_file()
    assert (run / "model_n5_ABCDF.png").is_file()
    import json
    s = json.loads((run / "summary.json").read_text())
    assert s["sweep"] is not None
    bts, btc = s["sweep"]["best_window_s"]
    assert 6.0 <= btc <= 12.0 and s["window_s"][1] == btc     # best cutoff used
