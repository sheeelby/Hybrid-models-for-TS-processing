"""Training loop utilities shared by different models."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

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

    model.train()
    for _ in range(cfg.epochs):
        for xb, yb in loader:
            xb = xb.to(device).float()
            yb = yb.to(device).float()

            optimizer.zero_grad()
            if hasattr(model, "forward_with_target"):
                preds = model.forward_with_target(xb, yb)
            else:
                preds = model(xb)
            loss = loss_fn(preds, yb)
            loss.backward()

            if cfg.clip is not None:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.clip)
            optimizer.step()

    model.eval()
    return model


__all__ = ["TrainConfig", "train_model"]
