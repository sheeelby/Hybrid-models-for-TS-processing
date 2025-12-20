"""Model factory functions for experiment scripts.

The pipelines can optionally pass per-model parameter overrides (via JSON),
so experiments can scale model capacity without code changes.
"""
from __future__ import annotations

from typing import Any, Mapping

import torch.nn as nn

from .nbeats import NBEATSV2
from .timesnet import TimesNetV2
from ..training.engine import TrainConfig


def _pick_params(
    defaults: Mapping[str, Any],
    overrides: Mapping[str, Any] | None,
) -> dict[str, Any]:
    params = dict(defaults)
    if overrides:
        for key, value in overrides.items():
            if key not in defaults:
                raise ValueError(f"Unsupported model param '{key}'. Allowed: {sorted(defaults)}")
            params[key] = value
    return params


def make_model(name: str, cfg: TrainConfig, *, params: Mapping[str, Any] | None = None) -> nn.Module:
    name = name.lower()
    if name == "timesnet":
        is_monthly = cfg.horizon >= 18
        defaults = {
            "d_model": 128 if is_monthly else 96,
            "layers": 6 if is_monthly else 5,
            "topk": 5,
            "dropout": 0.12,
            "use_anchor": True,
        }
        p = _pick_params(defaults, params)
        return TimesNetV2(cfg.lookback, cfg.horizon, **p)
    if name == "nbeats":
        is_monthly = cfg.horizon >= 18
        defaults = {
            "width": 768 if is_monthly else 640,
            "depth": 8 if is_monthly else 6,
            "nblocks": 12 if is_monthly else 10,
            "dropout": 0.15,
            "use_anchor": False,
            "share_weights": True,
            "use_trend": True,
            "use_seasonality": True,
            "use_generic": True,
            "diff_loss_weight": 0.08,
        }
        p = _pick_params(defaults, params)
        return NBEATSV2(cfg.lookback, cfg.horizon, **p)
    raise ValueError(f"Unknown model '{name}'")


__all__ = ["make_model"]
