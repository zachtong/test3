"""Tests for interrupted-seed detection: an interrupted seed leaves a best-val
checkpoint that must NOT be mistaken for a completed one (which would silently
ensemble an under-trained model)."""

from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import torch

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from training.checkpoint import history_len, seed_is_complete   # noqa: E402


def test_history_len(tmp_path):
    hp = tmp_path / "hist.npz"
    np.savez(hp, train=np.zeros(200), val=np.zeros(200))
    assert history_len(hp) == 200
    assert history_len(tmp_path / "missing.npz") == 0


def test_completed_seed_is_cached(tmp_path):
    cp = tmp_path / "model_seed7.pt"
    torch.save({}, cp)
    hp = tmp_path / "history_seed7.npz"
    np.savez(hp, train=np.zeros(200), val=np.zeros(200))
    assert seed_is_complete(cp, hp, epochs=200) is True


def test_interrupted_seed_is_not_cached(tmp_path):
    # best-val checkpoint exists, but history shows only 80 of 200 epochs
    cp = tmp_path / "model_seed27.pt"
    torch.save({}, cp)
    hp = tmp_path / "history_seed27.npz"
    np.savez(hp, train=np.zeros(80), val=np.zeros(80))
    assert seed_is_complete(cp, hp, epochs=200) is False


def test_missing_checkpoint_is_not_complete(tmp_path):
    assert seed_is_complete(tmp_path / "nope.pt", tmp_path / "h.npz", 200) is False


def test_checkpoint_without_history_assumed_complete(tmp_path):
    # legacy / externally supplied checkpoint with no history file
    cp = tmp_path / "model_seed7.pt"
    torch.save({}, cp)
    assert seed_is_complete(cp, tmp_path / "none.npz", epochs=200) is True
