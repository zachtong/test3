"""Train / val / test split."""

from __future__ import annotations

import numpy as np

from core.simulation import Simulation


def split_dataset(
    sims: list[Simulation],
    train_frac: float = 0.80,
    val_frac: float = 0.10,
    seed: int = 7,
    regimes: list[str] | None = None,
) -> dict:
    N = len(sims)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(N)
    n_train_total = int(N * train_frac)
    n_val = int(n_train_total * val_frac)
    n_train = n_train_total - n_val
    tr, vl, te = perm[:n_train], perm[n_train:n_train_total], perm[n_train_total:]
    result = dict(
        train_sims=[sims[i] for i in tr],
        val_sims=[sims[i] for i in vl],
        test_sims=[sims[i] for i in te],
        train_idx=tr, val_idx=vl, test_idx=te,
    )
    if regimes is not None:
        result["test_regimes"] = [regimes[i] for i in te]
    return result


def oversample_long(train_sims: list[Simulation], above: float | None,
                    factor: int) -> list[Simulation]:
    """Duplicate TRAIN sims whose real duration t_max exceeds ``above`` (s) so
    each appears ``factor`` times. Rare-regime rebalancing: a 0.2% share of
    long-recipe runs would otherwise be averaged away by the optimizer. Apply
    to the train list ONLY (never val/test), after the split and after the
    basis fit. Returns a new list; the input is not mutated.
    """
    if above is None or factor <= 1:
        return list(train_sims)
    long_tr = [s for s in train_sims
               if float(s.params.get("t_max", 0.0)) > above]
    return list(train_sims) + long_tr * (factor - 1)


def oversample_by_name_prefix(
    train_sims: list[Simulation], prefix: str | None, factor: int,
) -> tuple[list[Simulation], int]:
    """Duplicate TRAIN sims whose `params['basename']` starts with ``prefix``.
    The basename is the filename a sim was loaded from (e.g. the prefixed
    symlink name in mixed_npz). Pair this with a distinct --new-prefix on
    step1 to tag a fresh batch without touching the original NPZ files.

    Returns (new train list, number of sims that matched).
    """
    if not prefix or factor <= 1:
        return list(train_sims), 0
    matched = [s for s in train_sims
               if str(s.params.get("basename", "")).startswith(prefix)]
    return list(train_sims) + matched * (factor - 1), len(matched)


def oversample_by_source_substring(
    train_sims: list[Simulation], substring: str | None, factor: int,
) -> tuple[list[Simulation], int]:
    """Duplicate TRAIN sims whose source path contains ``substring`` so each
    appears ``factor`` times. Same train-only rebalancing pattern as
    oversample_long, but keyed on the per-NPZ ``source`` provenance string
    (set during JSON -> NPZ conversion). Use this to lift a newly added
    batch's share above the dataset noise floor when the batch isn't
    distinguishable by t_max or any other physical field.

    Returns (new train list, number of sims that matched the substring).
    """
    if not substring or factor <= 1:
        return list(train_sims), 0
    matched = [s for s in train_sims
               if substring in str(s.params.get("source", ""))]
    return list(train_sims) + matched * (factor - 1), len(matched)
