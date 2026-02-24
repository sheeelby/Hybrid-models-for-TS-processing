"""Direct (non-decomposed) neural forecasters for a single series."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import torch

from ..training import TrainConfig, WindowDatasetStd, train_model


@dataclass
class DirectNeuralForecaster:
    """Train a base neural model on the raw series (no MODWT) and forecast."""

    base_model_fn: Callable[[TrainConfig], torch.nn.Module]
    cfg: TrainConfig
    scale: bool = True

    model: Optional[torch.nn.Module] = None
    mu: float = 0.0
    sd: float = 1.0
    lookback: Optional[int] = None

    def fit(self, y) -> "DirectNeuralForecaster":
        y = np.asarray(y, float).ravel()
        H = int(self.cfg.horizon)
        min_windows = 8
        if y.size < max(2, H + 1):
            self.model = None
            self.lookback = None
            self.mu = float(np.mean(y)) if y.size else 0.0
            self.sd = float(np.std(y) + 1e-8) if y.size else 1.0
            return self

        Lmax = int(y.size - H)
        L = int(min(self.cfg.lookback, max(1, Lmax)))
        ds = WindowDatasetStd(y, L, H, stride=1, scale=bool(self.scale))
        if len(ds) < min_windows:
            self.model = None
            self.lookback = None
            self.mu = float(np.mean(y))
            self.sd = float(np.std(y) + 1e-8)
            return self

        self.mu, self.sd = ds.scaler
        self.lookback = L

        model = self.base_model_fn(self.cfg)
        self.model = train_model(model, ds, self.cfg)
        return self

    def forecast(self, y) -> np.ndarray:
        y = np.asarray(y, float).ravel()
        H = int(self.cfg.horizon)
        if y.size == 0:
            return np.zeros(H, dtype=float)
        if self.model is None or self.lookback is None or self.lookback <= 0:
            return np.repeat(float(y[-1]), H).astype(float)

        window = y[-self.lookback :]
        xb = ((window - float(self.mu)) / float(self.sd)).astype(np.float32).reshape(1, 1, -1)
        with torch.no_grad():
            pred = self.model(torch.from_numpy(xb).to(self.cfg.device)).detach().cpu().numpy().ravel()
        pred = pred * float(self.sd) + float(self.mu)
        if pred.size > H:
            pred = pred[:H]
        if pred.size < H:
            pred = np.pad(pred, (0, H - pred.size), mode="edge")
        return np.asarray(pred, float)


__all__ = ["DirectNeuralForecaster"]
