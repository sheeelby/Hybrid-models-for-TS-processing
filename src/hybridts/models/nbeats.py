"""Lightweight N-BEATS variant used in our experiments."""
from __future__ import annotations

import torch
import torch.nn as nn


class NBEATBlock(nn.Module):
    def __init__(self, lookback: int, horizon: int, width: int = 128, depth: int = 2):
        super().__init__()
        layers = []
        in_features = lookback
        for _ in range(depth):
            layers.extend([nn.Linear(in_features, width), nn.ReLU()])
            in_features = width
        self.fc = nn.Sequential(*layers)
        self.theta_b = nn.Linear(width, lookback)
        self.theta_f = nn.Linear(width, horizon)

    def forward(self, x: torch.Tensor):
        hidden = self.fc(x)
        return self.theta_b(hidden), self.theta_f(hidden)


class NBEATSV2(nn.Module):
    def __init__(self, lookback: int, horizon: int, width: int = 128, depth: int = 2, nblocks: int = 2):
        super().__init__()
        self.blocks = nn.ModuleList([
            NBEATBlock(lookback, horizon, width=width, depth=depth)
            for _ in range(nblocks)
        ])
        self.horizon = horizon

    def forward(self, x: torch.Tensor):
        batch, _, lookback = x.shape
        backcast = x.view(batch, lookback)
        forecast = torch.zeros(batch, self.horizon, device=x.device, dtype=x.dtype)
        for blk in self.blocks:
            b, f = blk(backcast)
            backcast = backcast - b
            forecast = forecast + f
        anchor = x[:, :, -1].squeeze(1).unsqueeze(1)
        return forecast + anchor


__all__ = ["NBEATSV2"]
