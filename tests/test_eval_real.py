"""Tests for the minimal 3D real-data evaluation: the channel config folds,
the quarter->full-disk mirror, Checkpoint-1 flags, and the end-to-end path."""

from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from core.grid import canonical_grid                          # noqa: E402
from core.sensors import (SensorConfig, place_sensors,        # noqa: E402
                          sensor_indices)
from models import create_model                               # noqa: E402
from data.real_experiment import real_config_from_yaml        # noqa: E402
import scripts.eval_real as ER                                # noqa: E402

_ABCDEF = ((0.52, 0.0), (0.52, 45.0), (0.52, 90.0),
           (0.847, 0.0), (0.847, 45.0), (0.847, 90.0))
_KEYS = ["w_XM", "w_DM", "w_YM", "w_XE", "w_DE", "w_YE"]


def test_real_exp_n6_config_folds_to_quarter():
    cfg = real_config_from_yaml(str(_root / "configs" / "real_exp_n6.yaml"))
    assert cfg.R == 0.15
    assert [c.theta_deg for c in cfg.channels] == [0, 45, 90, 0, 45, 90]
    assert [round(c.r_phys, 3) for c in cfg.channels] == \
        [0.078, 0.078, 0.078, 0.127, 0.127, 0.127]
    assert [c.key for c in cfg.channels] == _KEYS


def test_default_config_matches_the_yaml():
    y = real_config_from_yaml(str(_root / "configs" / "real_exp_n6.yaml"))
    d = ER._default_config()
    assert [(c.key, round(c.r_phys, 4), c.theta_deg) for c in d.channels] == \
        [(c.key, round(c.r_phys, 4), c.theta_deg) for c in y.channels]


def test_mirror_full_reflects_the_quarter():
    nx, ny, nt = 3, 3, 1
    w = np.arange(nx * ny, dtype=float).reshape(nx, ny, nt)
    x, y = canonical_grid(nx, ny)
    wf, xf, yf = ER._mirror_full(w, x, y)
    assert wf.shape == (2 * nx - 1, 2 * ny - 1, nt)
    assert xf[0] == -x[-1] and xf[nx - 1] == 0.0 and xf[-1] == x[-1]
    assert np.all(np.diff(xf) > 0) and np.all(np.diff(yf) > 0)
    # mirror symmetry about both axes: full[center+d] == full[center-d]
    c = nx - 1
    assert wf[c + 1, c, 0] == wf[c - 1, c, 0]     # x-axis mirror
    assert wf[c, c + 1, 0] == wf[c, c - 1, 0]     # y-axis mirror


def test_checkpoint1_flags_sign_and_units():
    bundle = {"sensor_rtheta": np.array(_ABCDEF, dtype=float)}
    cfg = ER._default_config()
    y = np.abs(np.random.default_rng(0).standard_normal((6, 20))) * 1e-6  # +ve
    w = np.random.default_rng(1).standard_normal((16, 16, 20)) * 1e-6
    checks = ER.checkpoint1(bundle, y, w, cfg)
    by = {n: (sev, ok) for n, sev, ok, _ in checks}
    assert by["sensor r in [0,1]"] == ("critical", True)
    assert by["field finite"] == ("critical", True)
    # all-positive inputs must TRIP the downward-sign advisory
    assert by["inputs mostly downward (negative)"] == ("advisory", False)


@pytest.fixture
def bundle_and_real(tmp_path):
    nx = ny = 24
    K = 6
    nt = 20
    n = 6
    x, y = canonical_grid(nx, ny)
    rng = np.random.default_rng(0)
    Phi, _ = np.linalg.qr(rng.standard_normal((nx * ny, K)))
    scfg = SensorConfig(n=n, strategy="custom", positions=_ABCDEF)
    sxy = place_sensors(scfg)
    m = create_model("bitcn", n_in=n, n_out=K, channels=8, dilations=(1, 2),
                     kernel=3, dropout=0.0, causal=False)
    bundle = dict(
        tag="synth", uses_front=False, Phi=Phi, sigma=np.linspace(5, 1, K),
        spatial_shape=np.array([nx, ny]), x_canon=x, y_canon=y,
        nx=nx, ny=ny, nt=nt, K=K, x_end=1.0, y_end=1.0,
        sensor_rtheta=np.array(_ABCDEF), sensor_xy=sxy,
        sensor_ij=sensor_indices(sxy, x, y),
        y_mean=np.zeros((1, n, 1)), y_std=np.ones((1, n, 1)),
        target_mean=np.zeros((1, K, 1)), target_std=np.ones((1, K, 1)),
        model=dict(arch="bitcn", n_in=n, n_out=K, channels=8,
                   dilations=[1, 2], kernel=3, dropout=0.0, causal=False),
        seeds=[7], state_dicts=[m.state_dict()])
    bp = tmp_path / "bundle.pt"
    torch.save(bundle, bp)
    T = 50
    raw = {"time": np.linspace(0, 13, T)}
    for i, k in enumerate(_KEYS):
        raw[k] = -(i + 1) * 1e-6 * (1 + np.sin(np.linspace(0, 3, T)))
    rp = tmp_path / "real.npz"
    np.savez(rp, **raw)
    return bp, rp


def test_eval_real_end_to_end(bundle_and_real, tmp_path):
    import subprocess
    bp, rp = bundle_and_real
    out = tmp_path / "eval"
    r = subprocess.run(
        [sys.executable, "scripts/eval_real.py", "--bundle", str(bp),
         "--real", str(rp), "--config",
         str(_root / "configs" / "real_exp_n6.yaml"), "--out-dir", str(out)],
        cwd=str(_root), capture_output=True, text=True)
    assert r.returncode in (0, 1), r.stderr[-800:]      # 1 only if a WARN-crit
    for f in ["real_inputs.png", "real_field.png", "field.npz", "summary.json"]:
        assert (out / f).is_file() and (out / f).stat().st_size > 0, f
    s = json.loads((out / "summary.json").read_text())
    assert s["w_shape"] == [24, 24, 20]
    crit = [c for c in s["checkpoint1"] if c["severity"] == "critical"]
    assert all(c["ok"] for c in crit)                    # critical checks pass
