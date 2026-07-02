"""Evaluation pipeline: models -> predictions -> field errors -> ResultSet.

POD-only: no front (rb) channel. Mirrors the 2D `scorer.evaluate_ensemble`
signature minus the basis-dispatch and front handling.
"""

from __future__ import annotations
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn

from core.simulation import Simulation
from core.pod_basis import PODBasis
from training.normalization import NormStats, invert_norm
from training.config import ExperimentConfig, config_to_dict
from evaluation.metrics import (relative_l2_error, per_mode_rel_l2, stats,
                                truncation_floor)
from evaluation.result import ResultSet


def predict_ensemble(models: list[nn.Module],
                     x_test: torch.Tensor) -> np.ndarray:
    preds = []
    for m in models:
        m.eval()
        with torch.no_grad():
            preds.append(m(x_test).cpu().numpy())
    return np.mean(np.stack(preds), axis=0)


def evaluate_from_cached_norms(
    models: list[nn.Module],
    x_test: torch.Tensor,
    a_test_true: np.ndarray,
    test_f_true_sq_norm: np.ndarray,
    test_f_perp_sq_norm: np.ndarray,
    test_basenames: list[str],
    basis_sigma: np.ndarray,
    target_stats: NormStats,
    cfg: ExperimentConfig,
) -> ResultSet:
    """Score without touching test_sims.f, using Parseval on an
    orthonormal POD basis:

      ||f_pred - f_true||^2 = ||a_pred - a_true||^2 + ||f_perp||^2
      ||f_true||^2           = ||a_true||^2 + ||f_perp||^2

    Precomputed once per (data, basis, split) tuple and stashed in the
    trajectory cache, so subsequent trains skip the 93 GB F reload.

    per_regime is empty here (regime metadata is not currently stored
    in the traj cache).
    """
    a_pred = invert_norm(predict_ensemble(models, x_test), target_stats)
    K = a_pred.shape[1]

    # Field errors via Parseval identity.
    a_diff_sq = np.sum((a_pred - a_test_true) ** 2, axis=(1, 2))
    field_err_sq = a_diff_sq + test_f_perp_sq_norm
    denom = np.maximum(test_f_true_sq_norm, 1e-24)
    field_errs = np.sqrt(field_err_sq / denom)
    floor_errs = np.sqrt(test_f_perp_sq_norm / denom)

    per_mode = {}
    for k in range(K):
        per_mode[f"a_{k+1}"] = stats(
            per_mode_rel_l2(a_pred[:, k, :], a_test_true[:, k, :]))
    per_sim_per_mode = {}
    for k in range(K):
        per_sim_per_mode[f"a_{k+1}"] = per_mode_rel_l2(
            a_pred[:, k, :], a_test_true[:, k, :]).tolist()

    per_seed_medians = []
    for m in models:
        m.eval()
        with torch.no_grad():
            p = invert_norm(m(x_test).cpu().numpy(), target_stats)
        a_diff_sq_m = np.sum((p - a_test_true) ** 2, axis=(1, 2))
        errs_m = np.sqrt(
            (a_diff_sq_m + test_f_perp_sq_norm) / denom)
        per_seed_medians.append(float(np.median(errs_m)))

    return ResultSet(
        config=config_to_dict(cfg),
        global_stats=stats(field_errs),
        per_regime={},
        per_mode=per_mode,
        truncation_floor=stats(floor_errs),
        gap_to_floor=(float(np.median(field_errs))
                       / max(float(np.median(floor_errs)), 1e-12)),
        n_params=sum(p.numel() for p in models[0].parameters()),
        per_seed_medians=per_seed_medians,
        per_sim_field_errs=field_errs.tolist(),
        per_sim_floor_errs=floor_errs.tolist(),
        per_sim_basenames=list(test_basenames),
        per_sim_per_mode_errs=per_sim_per_mode,
    )


def evaluate_ensemble(
    models: list[nn.Module],
    x_test: torch.Tensor,
    test_sims: list[Simulation],
    basis: PODBasis,
    target_stats: NormStats,
    cfg: ExperimentConfig,
    test_regimes: list[str] | None = None,
) -> ResultSet:
    a_pred = invert_norm(predict_ensemble(models, x_test), target_stats)
    K = a_pred.shape[1]

    field_errs = np.array([
        relative_l2_error(basis.reconstruct(a_pred[j]), test_sims[j].f)
        for j in range(len(test_sims))])

    a_true = basis.project_ensemble(test_sims)
    per_mode: dict[str, dict] = {}
    for k in range(K):
        per_mode[f"a_{k+1}"] = stats(
            per_mode_rel_l2(a_pred[:, k, :], a_true[:, k, :]))

    per_regime: dict[str, dict] = {}
    if test_regimes:
        bucket: dict[str, list] = defaultdict(list)
        for e, tag in zip(field_errs, test_regimes):
            bucket[tag].append(e)
        per_regime = {t: stats(np.array(v)) for t, v in bucket.items()}

    floor_errs = truncation_floor(basis, test_sims)

    per_seed_medians = []
    for m in models:
        m.eval()
        with torch.no_grad():
            p = invert_norm(m(x_test).cpu().numpy(), target_stats)
        recs = [basis.reconstruct(p[j]) for j in range(len(test_sims))]
        errs = np.array([relative_l2_error(recs[j], test_sims[j].f)
                         for j in range(len(test_sims))])
        per_seed_medians.append(float(np.median(errs)))

    per_sim_per_mode = {}
    for k in range(K):
        per_sim_per_mode[f"a_{k+1}"] = per_mode_rel_l2(
            a_pred[:, k, :], a_true[:, k, :]).tolist()

    return ResultSet(
        config=config_to_dict(cfg),
        global_stats=stats(field_errs),
        per_regime=per_regime,
        per_mode=per_mode,
        truncation_floor=stats(floor_errs),
        gap_to_floor=float(np.median(field_errs))
                     / max(float(np.median(floor_errs)), 1e-12),
        n_params=sum(p.numel() for p in models[0].parameters()),
        per_seed_medians=per_seed_medians,
        per_sim_field_errs=field_errs.tolist(),
        per_sim_floor_errs=floor_errs.tolist(),
        per_sim_basenames=[s.params.get("basename", "")
                           for s in test_sims],
        per_sim_per_mode_errs=per_sim_per_mode,
    )
