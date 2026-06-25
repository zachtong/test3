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
    )
