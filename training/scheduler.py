"""LR schedule factory."""

from __future__ import annotations

import torch.optim as optim
from torch.optim.lr_scheduler import (
    CosineAnnealingLR, OneCycleLR, ReduceLROnPlateau,
    SequentialLR, LinearLR,
)

from training.config import TrainConfig


def make_scheduler(optimizer: optim.Optimizer, cfg: TrainConfig,
                   steps_per_epoch: int = 1):
    if cfg.lr_schedule == "cosine":
        return CosineAnnealingLR(optimizer, T_max=cfg.epochs,
                                 eta_min=cfg.lr_final)
    if cfg.lr_schedule == "onecycle":
        return OneCycleLR(optimizer, max_lr=cfg.lr_init,
                          total_steps=cfg.epochs * steps_per_epoch)
    if cfg.lr_schedule == "plateau":
        return ReduceLROnPlateau(optimizer, mode="min",
                                 patience=20, factor=0.5)
    if cfg.lr_schedule == "warmup_cosine":
        warmup_epochs = min(10, cfg.epochs // 10)
        warmup = LinearLR(optimizer, start_factor=0.01,
                          total_iters=warmup_epochs)
        cosine = CosineAnnealingLR(optimizer,
                                   T_max=cfg.epochs - warmup_epochs,
                                   eta_min=cfg.lr_final)
        return SequentialLR(optimizer, schedulers=[warmup, cosine],
                            milestones=[warmup_epochs])
    raise ValueError(f"unknown lr_schedule: {cfg.lr_schedule!r}")
