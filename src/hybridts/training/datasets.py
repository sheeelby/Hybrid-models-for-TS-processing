"""Dataset helpers used during neural network training."""
from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


class WindowDatasetStd(Dataset):
    """Standard sliding window dataset with optional z-score scaling."""

    def __init__(self, y, lookback: int, horizon: int, stride: int = 1, scale: bool = True):
        y = np.asarray(y, np.float32)
        if scale:
            mu = float(y.mean())
            sd = float(y.std()) + 1e-8
            z = (y - mu) / sd
        else:
            mu, sd, z = 0.0, 1.0, y
        X, Y = [], []
        for t in range(0, len(z) - lookback - horizon + 1, stride):
            X.append(z[t : t + lookback])
            Y.append(z[t + lookback : t + lookback + horizon])
        self.X = np.asarray(X, np.float32)
        self.Y = np.asarray(Y, np.float32)
        self._scaler = (mu, sd)

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.X)

    def __getitem__(self, idx: int):  # pragma: no cover - thin wrapper
        return (
            torch.from_numpy(self.X[idx]).unsqueeze(0),
            torch.from_numpy(self.Y[idx]),
        )

    @property
    def scaler(self) -> Tuple[float, float]:
        return self._scaler


__all__ = ["WindowDatasetStd"]
