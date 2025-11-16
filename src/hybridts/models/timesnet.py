"""TimesNet-inspired forecaster used in hybrid experiments."""
from __future__ import annotations

import torch
import torch.nn as nn

from .timesnet_blocks import MSUnit


class TimesBlockV2(nn.Module):
    def __init__(
        self,
        lookback: int,
        horizon: int,
        d_model: int = 32,
        topk: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.horizon = horizon
        self.topk = topk
        self.msunit = MSUnit(1, d_model, dropout=dropout)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lookback, horizon),
        )

    def _reshape_2d(self, x: torch.Tensor, period: int):
        batch, _, length = x.shape
        cols = (length + period - 1) // period
        total = cols * period
        pad = total - length
        if pad > 0:
            pad_tensor = torch.zeros(batch, 1, pad, device=x.device, dtype=x.dtype)
            x = torch.cat([x, pad_tensor], dim=-1)
        reshaped = x.view(batch, 1, cols, period).transpose(-1, -2).contiguous()
        return reshaped, pad, length

    def _reshape_back(self, x2d: torch.Tensor, pad: int, original_len: int):
        batch, channels, period, freq = x2d.shape
        flattened = x2d.transpose(-1, -2).contiguous().view(batch, channels, period * freq)
        if pad > 0:
            flattened = flattened[..., :-pad]
        if flattened.shape[-1] != original_len:
            flattened = flattened[..., :original_len]
        return flattened

    def forward(self, x: torch.Tensor):
        series_len = x.shape[-1]
        amp = torch.fft.rfft(x[:, 0, :], dim=-1).abs().mean(0)[1:]
        k = int(min(self.topk, amp.numel()))
        if k <= 0:
            period_list = [series_len]
            weights = torch.tensor([1.0], device=x.device, dtype=x.dtype)
        else:
            vals, idx = torch.topk(amp, k)
            freqs = idx + 1
            period_list = [max(1, int(round(series_len / int(freq.item())))) for freq in freqs]
            weights = torch.softmax(vals.to(x.device, dtype=x.dtype), dim=0)

        outputs = []
        for period in period_list:
            x2d, pad, original_len = self._reshape_2d(x, period)
            y2d = self.msunit(x2d)
            outputs.append(self._reshape_back(y2d, pad, original_len))

        combined = torch.zeros_like(x)
        for w, out in zip(weights, outputs):
            combined = combined + w * out

        flat = combined.squeeze(1)
        return self.head(flat)


class TimesNetV2(nn.Module):
    def __init__(
        self,
        lookback: int,
        horizon: int,
        d_model: int = 32,
        layers: int = 2,
        topk: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                TimesBlockV2(lookback, horizon, d_model=d_model, topk=topk, dropout=dropout)
                for _ in range(layers)
            ]
        )
        self.horizon = horizon

    def forward(self, x: torch.Tensor):
        output = None
        states = x
        for block in self.blocks:
            block_out = block(states)
            output = block_out if output is None else output + block_out
        anchor = x[:, :, -1].squeeze(1).unsqueeze(1)
        return output + anchor


__all__ = ["TimesNetV2"]
