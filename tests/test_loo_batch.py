"""Test batch leave-one-out over a folder of CSVs."""

from __future__ import annotations
import json
import subprocess
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


def _mk_bundle(positions, path, nx=32, ny=32, K=6, nt=20):
    x, y = canonical_grid(nx, ny)
    Phi, _ = np.linalg.qr(np.random.default_rng(0).standard_normal((nx * ny, K)))
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


def _mk_csv(path, scale):
    rows = ["Zero position[nm]", ",XM,XE,YM,YE,DM,DE", ",10,20,10,20,10,20",
            "Sampling Data[nm]", "Time[ms],XM,XE,YM,YE,DM,DE"]
    for k in range(40):
        t = k * 300
        vals = [-(i + 1) * scale * (1 - np.exp(-t / 4000)) for i in range(6)]
        rows.append(",".join([str(t)] + [f"{v:.1f}" for v in vals]))
    path.write_text("\n".join(rows) + "\n")


def test_batch_loo_over_folder(tmp_path):
    bd = tmp_path / "b"
    bd.mkdir()
    _mk_bundle([(0.52, 0), (0.52, 45), (0.52, 90), (0.847, 0), (0.847, 45)],
               bd / "merged_sweep_k12_n5_ABCDE.pt")
    cd = tmp_path / "csvs"
    cd.mkdir()
    for nm, sc in [("run_A", 100.0), ("run_B", 200.0)]:
        _mk_csv(cd / f"{nm}.csv", sc)
    out = tmp_path / "out"
    r = subprocess.run(
        [sys.executable, "run_loo_batch.py", "--csv-dir", str(cd),
         "--bundles", str(bd / "merged_sweep_k12_n5_ABCDE.pt"),
         "--config", str(_root / "configs" / "real_exp_n6.yaml"),
         "--auto-cutoff", "--out-dir", str(out)],
        cwd=str(_root), capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-800:]
    for nm in ("run_A", "run_B"):                              # per-run folders
        assert (out / nm / "summary.json").is_file()
        assert (out / nm / "loo.png").is_file()
    b = json.loads((out / "batch_loo_summary.json").read_text())
    assert b["n_runs"] == 2
    assert (out / "batch_loo_summary.png").is_file()

    # --skip-existing: a second run re-uses the summaries, re-runs nothing
    r2 = subprocess.run(
        [sys.executable, "run_loo_batch.py", "--csv-dir", str(cd),
         "--bundles", str(bd / "merged_sweep_k12_n5_ABCDE.pt"),
         "--config", str(_root / "configs" / "real_exp_n6.yaml"),
         "--auto-cutoff", "--out-dir", str(out), "--skip-existing"],
        cwd=str(_root), capture_output=True, text=True)
    assert r2.returncode == 0
    assert "[skip]" in r2.stdout
