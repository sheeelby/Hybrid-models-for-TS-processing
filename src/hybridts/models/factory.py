"""Model factory functions for experiment scripts."""
from __future__ import annotations

import torch.nn as nn

from .helformer import HelformerAutoRegressor
from .nbeats import NBEATSV2
from .timesnet import TimesNetV2
from ..training.engine import TrainConfig


def make_model(name: str, cfg: TrainConfig) -> nn.Module:
    name = name.lower()
    if name == "timesnet":
        # Slightly regularised TimesNet variant for small datasets
        return TimesNetV2(cfg.lookback, cfg.horizon, d_model=32, layers=2, topk=2, dropout=0.1)
    if name == "nbeats":
        # Narrower N-BEATS with dropout to reduce overfitting on short series
        return NBEATSV2(cfg.lookback, cfg.horizon, width=64, depth=2, nblocks=2, dropout=0.1)
    if name == "helformer":
        # Compact Helformer configuration tuned for limited data
        return HelformerAutoRegressor(
            horizon=cfg.horizon,
            input_dim=1,
            num_heads=2,
            head_dim=16,
            lstm_units=16,
            dropout=0.1,
        )
    raise ValueError(f"Unknown model '{name}'")


__all__ = ["make_model"]
