"""Tests for the self-contained 3D inference bundle + standalone reconstruct.

Fabricates a fake trained run (results.json + norm_stats + checkpoints + a
pod3d basis file) in a tmp dir, so no dataset or real training is needed."""

from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from training.config import (ExperimentConfig, DataConfig, PODConfig,   # noqa: E402
                             SensorConfig, ModelConfig, config_to_dict)
from training.normalization import NormStats, save_norm_stats           # noqa: E402
from training.checkpoint import checkpoint_path                         # noqa: E402
from evaluation.result import ResultSet                                 # noqa: E402
from models import create_model                                         # noqa: E402
from scripts.bundle import build_bundle                                 # noqa: E402
from scripts.reconstruct import load_bundle, reconstruct_field          # noqa: E402
from data.real_experiment import (Channel, RealExperimentConfig,        # noqa: E402
                                  assemble_inputs, fold_theta)

_NX = _NY = 16
_K = 6
_NT = 20
_N = 6
_SEEDS = (7, 17)
_ABCDEF = ((0.52, 0.0), (0.52, 45.0), (0.52, 90.0),
           (0.847, 0.0), (0.847, 45.0), (0.847, 90.0))


@pytest.fixture
def bundle(tmp_path):
    """Fabricate a fake run and build its bundle; return the loaded bundle."""
    out_dir = tmp_path / "outputs"
    tag = "synthtag"
    (out_dir / tag).mkdir(parents=True)

    cfg = ExperimentConfig(
        data=DataConfig(nx=_NX, ny=_NY, nt=_NT, x_end=1.0, y_end=1.0),
        pod=PODConfig(k=_K, k_cache=_K),
        sensors=SensorConfig(n=_N, strategy="custom", positions=_ABCDEF),
        model=ModelConfig(arch="bitcn", channels=8, kernel=3,
                          dilations=(1, 2), dropout=0.0, causal=False),
        seeds=_SEEDS, output_dir=str(out_dir), tag=tag)
    ResultSet(config=config_to_dict(cfg), global_stats={}, per_regime={},
              per_mode={}, truncation_floor={}, gap_to_floor=0.0,
              n_params=0, per_seed_medians=[]).save_json(
                  out_dir / tag / "results.json")

    save_norm_stats(out_dir / tag / "norm_stats.npz",
                    NormStats(np.zeros((1, _N, 1)), np.ones((1, _N, 1))),
                    NormStats(np.zeros((1, _K, 1)), np.ones((1, _K, 1))))
    for sd in _SEEDS:
        m = create_model("bitcn", n_in=_N, n_out=_K, channels=8,
                         dilations=(1, 2), kernel=3, dropout=0.0, causal=False)
        torch.save(m.state_dict(), checkpoint_path(str(out_dir), tag, sd))

    rng = np.random.default_rng(0)
    Phi, _ = np.linalg.qr(rng.standard_normal((_NX * _NY, _K)))
    basis_file = tmp_path / "pod3d_test.npz"
    np.savez(basis_file, Phi=Phi, sigma=np.linspace(5, 1, _K),
             spatial_shape=np.array([_NX, _NY]), k_cache=_K)

    out_bundle = tmp_path / "bundle.pt"
    build_bundle(tag, str(basis_file), str(out_bundle), output_dir=str(out_dir))
    return load_bundle(out_bundle)


def test_bundle_is_pod_only_with_expected_metadata(bundle):
    assert bundle["uses_front"] is False
    assert tuple(bundle["spatial_shape"]) == (_NX, _NY)
    assert bundle["K"] == _K and bundle["nt"] == _NT
    assert bundle["Phi"].shape == (_NX * _NY, _K)
    assert bundle["model"]["n_out"] == _K            # no +1 front channel
    assert len(bundle["state_dicts"]) == len(_SEEDS)


def test_sensor_positions_folded_into_quarter(bundle):
    rtheta = np.asarray(bundle["sensor_rtheta"])
    assert rtheta.shape == (_N, 2)
    assert np.all((rtheta[:, 1] >= 0.0) & (rtheta[:, 1] <= 90.0))
    ij = np.asarray(bundle["sensor_ij"])
    assert ij.shape == (_N, 2)
    assert ij.min() >= 0 and ij.max() < _NX


def test_reconstruct_shape_and_finiteness(bundle):
    y = np.random.default_rng(1).standard_normal((_N, _NT))
    w = reconstruct_field(bundle, y)
    assert w.shape == (_NX, _NY, _NT)
    assert np.all(np.isfinite(w))


def test_reconstruct_resamples_offgrid_time(bundle):
    T = 37                                            # != nt, forces resample
    t_raw = np.linspace(0.0, 13.0, T)
    y = np.random.default_rng(2).standard_normal((_N, T))
    w = reconstruct_field(bundle, y, t_raw=t_raw)
    assert w.shape == (_NX, _NY, _NT)


def test_full_chain_real_npz_to_field(bundle):
    T = 50
    raw = {"time": np.linspace(0.0, 13.0, T)}
    keys = ["w_XM", "w_DM", "w_YM", "w_XE", "w_DE", "w_YE"]
    for i, k in enumerate(keys):
        raw[k] = (i + 1) * np.sin(np.linspace(0, 3, T))
    chans = (
        Channel("w_XM", 0.078, fold_theta(180)),
        Channel("w_DM", 0.078, fold_theta(-45)),
        Channel("w_YM", 0.078, fold_theta(90)),
        Channel("w_XE", 0.127, fold_theta(180)),
        Channel("w_DE", 0.127, fold_theta(-45)),
        Channel("w_YE", 0.127, fold_theta(90)),
    )
    cfg = RealExperimentConfig(R=0.15, channels=chans, t_cutoff=13.0)
    y, t = assemble_inputs(raw, bundle["sensor_rtheta"], cfg)
    w = reconstruct_field(bundle, y, t_raw=t)
    assert w.shape == (_NX, _NY, _NT)
    assert np.all(np.isfinite(w))
