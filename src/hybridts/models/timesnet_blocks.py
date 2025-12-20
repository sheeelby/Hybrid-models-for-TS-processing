
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MSUnit(nn.Module):
    def __init__(self, in_ch: int = 1, hidden_ch: int = 32, dropout: float = 0.1):
        super().__init__()
        self.act = nn.GELU()
        self.k1 = nn.Conv2d(in_ch, hidden_ch, kernel_size=1, padding=0, bias=False)
        self.k3 = nn.Conv2d(in_ch, hidden_ch, kernel_size=3, padding=1, bias=False)
        self.k5 = nn.Conv2d(in_ch, hidden_ch, kernel_size=5, padding=2, bias=False)
        self.k7 = nn.Conv2d(in_ch, hidden_ch, kernel_size=7, padding=3, bias=False)
        self.pool = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_ch, hidden_ch, kernel_size=1, bias=False),
        )
        self.norm = nn.GroupNorm(1, 5 * hidden_ch)
        self.dropout = nn.Dropout2d(dropout)
        self.proj = nn.Conv2d(5 * hidden_ch, in_ch, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.act(self.k1(x))
        b = self.act(self.k3(x))
        c = self.act(self.k5(x))
        d = self.act(self.k7(x))
        e = self.act(self.pool(x))
        y = torch.cat([a, b, c, d, e], dim=1)
        y = self.norm(y)
        y = self.dropout(y)
        return self.proj(y)


__all__ = ["MSUnit"]
