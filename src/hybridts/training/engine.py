"""Training loop utilities shared by different models."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

try:  # pragma: no cover - optional dependency
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - fallback
    tqdm = None

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


@dataclass
class TrainConfig:
    lookback: int
    horizon: int
    epochs: int = 10
    batch_size: int = 64
    lr: float = 3e-3
    weight_decay: float = 1e-4
    clip: Optional[float] = 1.0
    device: Optional[str] = None
    # Optional auxiliary loss weights (used only if a model exposes matching attributes).
    # Kept in TrainConfig so pipelines can control behaviour without changing model code.
    diff_loss_weight: float = 0.0

    def __post_init__(self) -> None:
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"


def train_model(model: torch.nn.Module, dataset: Dataset, cfg: TrainConfig):
    if len(dataset) == 0:
        return None

    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)
    device = torch.device(cfg.device)
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_fn = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, cfg.epochs))

    epoch_iter = range(cfg.epochs)
    if tqdm is not None:
        epoch_iter = tqdm(epoch_iter, desc=f"train ({model.__class__.__name__})", leave=False)

    model.train()
    for _ in epoch_iter:
        for xb, yb in loader:
            xb = xb.to(device).float()
            yb = yb.to(device).float()

            optimizer.zero_grad()
            if hasattr(model, "forward_with_target"):
                preds = model.forward_with_target(xb, yb)
            else:
                preds = model(xb)
            loss = loss_fn(preds, yb)

            # Smoothness / anti-spike regularization.
            # If enabled, match first differences between prediction and target.
            w = float(getattr(model, "diff_loss_weight", cfg.diff_loss_weight))
            if w > 0 and preds.ndim == 2 and yb.ndim == 2 and preds.size(1) >= 2:
                dp = preds[:, 1:] - preds[:, :-1]
                dt = yb[:, 1:] - yb[:, :-1]
                loss = loss + w * loss_fn(dp, dt)
            loss.backward()

            if cfg.clip is not None:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.clip)
            optimizer.step()
        scheduler.step()

    model.eval()
    return model


__all__ = ["TrainConfig", "train_model"]
