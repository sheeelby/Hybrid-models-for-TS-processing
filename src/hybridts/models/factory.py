"""Model factory functions for experiment scripts."""
from __future__ import annotations

import torch.nn as nn

from .nbeats import NBEATSV2
from .timesnet import TimesNetV2
from ..training.engine import TrainConfig


def make_model(name: str, cfg: TrainConfig) -> nn.Module:
    name = name.lower()
    if name == "timesnet":
        return TimesNetV2(cfg.lookback, cfg.horizon, d_model=32, layers=2, topk=2)
    if name == "nbeats":
        return NBEATSV2(cfg.lookback, cfg.horizon, width=128, depth=2, nblocks=2)
    raise ValueError(f"Unknown model '{name}'")


__all__ = ["make_model"]
