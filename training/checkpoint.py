"""Checkpoint save/load."""

from __future__ import annotations
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from training.config import ExperimentConfig, config_to_dict


def config_hash(cfg: ExperimentConfig) -> str:
    blob = json.dumps(config_to_dict(cfg), sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()[:12]


def checkpoint_path(output_dir: str | Path, tag: str, seed: int) -> Path:
    d = Path(output_dir) / tag / "checkpoints"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"model_seed{seed}.pt"


def history_path(output_dir: str | Path, tag: str, seed: int) -> Path:
    d = Path(output_dir) / tag / "checkpoints"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"history_seed{seed}.npz"


def latest_path(output_dir: str | Path, tag: str, seed: int) -> Path:
    """Rolling mid-training snapshot (for crash/interrupt recovery)."""
    d = Path(output_dir) / tag / "checkpoints"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"latest_seed{seed}.pt"


def save_checkpoint(model: nn.Module, history: dict,
                    path: Path, hist_path: Path) -> None:
    torch.save(model.state_dict(), path)
    np.savez(hist_path, train=np.array(history["train"]),
             val=np.array(history["val"]))


def load_checkpoint(model: nn.Module, path: Path,
                    hist_path: Path, device: torch.device) -> dict:
    model.load_state_dict(torch.load(path, map_location=device))
    if hist_path.exists():
        z = np.load(hist_path)
        return {"train": z["train"].tolist(), "val": z["val"].tolist()}
    return {"train": [], "val": []}
