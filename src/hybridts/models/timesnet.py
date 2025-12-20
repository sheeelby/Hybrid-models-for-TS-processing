from __future__ import annotations

import torch
import torch.nn as nn

from .timesnet_blocks import MSUnit


class TimesBlockV2(nn.Module):
    def __init__(
        self,
        lookback: int,
        d_model: int,
        topk: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.lookback = int(lookback)
        self.d_model = int(d_model)
        self.topk = int(topk)
        hidden_ch = max(16, min(64, self.d_model // 2))
        self.msunit = MSUnit(self.d_model, hidden_ch=hidden_ch, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(self.d_model)
        self.norm2 = nn.LayerNorm(self.d_model)
        ffn_hidden = max(128, 4 * self.d_model)
        self.ffn = nn.Sequential(
            nn.Conv1d(self.d_model, ffn_hidden, kernel_size=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(ffn_hidden, self.d_model, kernel_size=1),
            nn.Dropout(dropout),
        )

    def _ln(self, x: torch.Tensor, norm: nn.LayerNorm) -> torch.Tensor:
        return norm(x.transpose(1, 2)).transpose(1, 2)

    def _reshape_2d(self, x: torch.Tensor, period: int):
        batch, channels, length = x.shape
        cols = (length + period - 1) // period
        total = cols * period
        pad = total - length
        if pad > 0:
            last = x[:, :, -1:].expand(batch, channels, pad)
            x = torch.cat([x, last.to(device=x.device, dtype=x.dtype)], dim=-1)
        reshaped = x.view(batch, channels, cols, period).transpose(-1, -2).contiguous()
        return reshaped, pad, length

    def _reshape_back(self, x2d: torch.Tensor, pad: int, original_len: int):
        batch, channels, period, freq = x2d.shape
        flattened = x2d.transpose(-1, -2).contiguous().view(batch, channels, period * freq)
        if pad > 0:
            flattened = flattened[..., :-pad]
        if flattened.shape[-1] != original_len:
            flattened = flattened[..., :original_len]
        return flattened

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_with_periods(x, period_list=None, weights=None)

    def forward_with_periods(
        self,
        x: torch.Tensor,
        *,
        period_list: list[int] | None,
        weights: torch.Tensor | None,
    ) -> torch.Tensor:
        series_len = x.shape[-1]
        if period_list is None or weights is None:
            period_list = [series_len]
            weights = torch.ones(x.size(0), 1, device=x.device, dtype=x.dtype)

        outputs = []
        for period in period_list:
            x2d, pad, original_len = self._reshape_2d(x, period)
            y2d = self.msunit(x2d)
            outputs.append(self._reshape_back(y2d, pad, original_len))

        combined = torch.zeros_like(x)
        for i, out in enumerate(outputs):
            combined = combined + weights[:, i].view(-1, 1, 1) * out

        y = self._ln(x + self.dropout(combined), self.norm1)
        y = self._ln(y + self.ffn(y), self.norm2)
        return y


class TimesNetV2(nn.Module):
    def __init__(
        self,
        lookback: int,
        horizon: int,
        d_model: int = 64,
        layers: int = 4,
        topk: int = 3,
        dropout: float = 0.1,
        use_anchor: bool = True,
    ) -> None:
        super().__init__()
        self.lookback = int(lookback)
        self.horizon = int(horizon)
        self.d_model = int(d_model)
        self.topk = int(topk)
        self.use_anchor = bool(use_anchor)

        self.in_proj = nn.Conv1d(1, self.d_model, kernel_size=1)
        self.blocks = nn.ModuleList(
            [
                TimesBlockV2(
                    lookback=self.lookback,
                    d_model=self.d_model,
                    topk=self.topk,
                    dropout=dropout,
                )
                for _ in range(int(layers))
            ]
        )
        self.out_proj = nn.Conv1d(self.d_model, 1, kernel_size=1)

        head_hidden = max(128, 2 * self.d_model)
        self.head = nn.Sequential(
            nn.Linear(self.lookback, head_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, head_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, self.horizon),
        )

    def _periods(self, x_raw: torch.Tensor) -> tuple[list[int], torch.Tensor]:
        series_len = x_raw.shape[-1]
        amp = torch.fft.rfft(x_raw, dim=-1).abs()
        amp_nz = amp[:, 1:]
        if amp_nz.numel() == 0:
            return [series_len], torch.ones(x_raw.size(0), 1, device=x_raw.device, dtype=x_raw.dtype)
        amp_mean = amp_nz.mean(0)
        k = int(min(self.topk, amp_mean.numel()))
        if k <= 0:
            return [series_len], torch.ones(x_raw.size(0), 1, device=x_raw.device, dtype=x_raw.dtype)
        _, idx = torch.topk(amp_mean, k)
        freqs = idx + 1
        period_list = [max(1, int(round(series_len / int(freq.item())))) for freq in freqs]
        weights = torch.softmax(amp_nz[:, idx].to(x_raw.device, dtype=x_raw.dtype), dim=-1)
        return period_list, weights

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        period_list, weights = self._periods(x[:, 0, :])
        states = self.in_proj(x)
        for block in self.blocks:
            states = block.forward_with_periods(states, period_list=period_list, weights=weights)
        states = self.out_proj(states)
        out = self.head(states.squeeze(1))
        if not self.use_anchor:
            return out
        anchor = x[:, :, -1].squeeze(1).unsqueeze(1)
        return out + anchor


__all__ = ["TimesNetV2"]