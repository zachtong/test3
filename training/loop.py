"""Training loop."""

from __future__ import annotations
import time

import numpy as np
import torch
import torch.nn as nn

from training.config import TrainConfig
from training.loss import channel_weighted_mse
from training.scheduler import make_scheduler


def train_one_seed(
    model: nn.Module,
    x_train: torch.Tensor, y_train: torch.Tensor,
    x_val: torch.Tensor, y_val: torch.Tensor,
    weights: torch.Tensor, cfg: TrainConfig,
    device: torch.device, seed: int,
    verbose: bool = True,
    save_fn=None, best_path=None, latest_path=None, hist_path=None,
) -> tuple[nn.Module, dict]:
    """Train one seed.

    If save_fn/best_path are given, the best-val weights are written to disk the
    moment val improves (so the best is never lost to a crash), and a rolling
    'latest' snapshot is written every cfg.checkpoint_every epochs.
    """
    can_save = save_fn is not None and hist_path is not None
    torch.manual_seed(seed)
    np.random.seed(seed)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr_init, weight_decay=cfg.weight_decay)
    steps_per_epoch = max(1, len(x_train) // cfg.batch_size)
    scheduler = make_scheduler(optimizer, cfg, steps_per_epoch)

    history: dict[str, list[float]] = {"train": [], "val": []}
    best_val = float("inf")
    best_state = None
    t0 = time.time()

    for ep in range(cfg.epochs):
        model.train()
        order = torch.randperm(len(x_train), device=device)
        ep_loss, nb = 0.0, 0

        for i in range(0, len(x_train), cfg.batch_size):
            idx = order[i:i + cfg.batch_size]
            optimizer.zero_grad()
            loss = channel_weighted_mse(model(x_train[idx]),
                                        y_train[idx], weights)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                           max_norm=cfg.grad_clip)
            optimizer.step()
            if cfg.lr_schedule == "onecycle":
                scheduler.step()
            ep_loss += loss.item()
            nb += 1

        if cfg.lr_schedule != "onecycle":
            if cfg.lr_schedule == "plateau":
                scheduler.step(ep_loss / max(nb, 1))
            else:
                scheduler.step()

        tr_loss = ep_loss / max(nb, 1)
        model.eval()
        with torch.no_grad():
            val_loss = channel_weighted_mse(
                model(x_val), y_val, weights).item()

        history["train"].append(tr_loss)
        history["val"].append(val_loss)
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().clone()
                          for k, v in model.state_dict().items()}
            if can_save and best_path is not None:
                save_fn(model, history, best_path, hist_path)   # persist best now

        if can_save and latest_path is not None and cfg.checkpoint_every > 0 \
                and (ep + 1) % cfg.checkpoint_every == 0:
            save_fn(model, history, latest_path, hist_path)     # rolling snapshot

        if verbose and ((ep + 1) % cfg.print_every == 0 or ep == 0):
            print(f"    ep {ep+1:3d}/{cfg.epochs}  "
                  f"train={tr_loss:.3e}  val={val_loss:.3e}  "
                  f"t={time.time()-t0:.0f}s")

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history
