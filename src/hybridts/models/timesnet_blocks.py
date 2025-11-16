"""Building blocks for the simplified TimesNet implementation."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MSUnit(nn.Module):
    def __init__(self, in_ch: int = 1, d_model: int = 32, dropout: float = 0.1):
        super().__init__()
        self.act = nn.GELU()
        self.p3 = nn.Conv2d(in_ch, d_model, kernel_size=(3, 1), padding=0, bias=False)
        self.p5 = nn.Conv2d(in_ch, d_model, kernel_size=(5, 1), padding=0, bias=False)
        self.f3 = nn.Conv2d(in_ch, d_model, kernel_size=(1, 3), padding=0, bias=False)
        self.f5 = nn.Conv2d(in_ch, d_model, kernel_size=(1, 5), padding=0, bias=False)
        self.proj = nn.Conv2d(4 * d_model, in_ch, kernel_size=1, bias=False)
        self.norm = nn.GroupNorm(1, 4 * d_model)
        # Spatial dropout over feature maps to improve generalisation on small datasets.
        self.dropout = nn.Dropout2d(dropout)

    def _causal_T(self, conv: nn.Conv2d, x: torch.Tensor) -> torch.Tensor:
        kT = conv.kernel_size[0]
        x_pad = F.pad(x, (0, 0, kT - 1, 0))
        return conv(x_pad)

    def _causal_F(self, conv: nn.Conv2d, x: torch.Tensor) -> torch.Tensor:
        kF = conv.kernel_size[1]
        x_pad = F.pad(x, (kF - 1, 0, 0, 0))
        return conv(x_pad)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.act(self._causal_T(self.p3, x))
        b = self.act(self._causal_T(self.p5, x))
        c = self.act(self._causal_F(self.f3, x))
        d = self.act(self._causal_F(self.f5, x))
        y = torch.cat([a, b, c, d], dim=1)
        y = self.norm(y)
        y = self.dropout(y)
        return self.proj(y)


__all__ = ["MSUnit"]
