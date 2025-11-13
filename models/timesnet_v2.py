# models/timesnet_v2.py
import torch
import torch.nn as nn
from .timesnet_blocks import MSUnit

class TimesBlockV2(nn.Module):
    def __init__(self, lookback, horizon, d_model=32, topk=2):
        super().__init__()
        self.horizon = horizon
        self.topk = topk
        self.msunit = MSUnit(1, d_model)
        self.head = nn.Linear(lookback, horizon)  # проекция по времени T -> H

    def _reshape_2d(self, x, P):
        B, _, T = x.shape
        cols = (T + P - 1) // P
        Tpad = P * cols
        pad = Tpad - T
        if pad > 0:
            x = torch.cat([x, torch.zeros(B, 1, pad, device=x.device, dtype=x.dtype)], dim=-1)
        # (B,1,T) -> (B,1,cols,P) и меняем оси, чтобы P стало "высотой"
        return x.view(B, 1, cols, P).transpose(-1, -2).contiguous(), pad, T

    def _reshape_back(self, x2d, pad, T):
        B, C, P, F = x2d.shape
        x = x2d.transpose(-1, -2).contiguous().view(B, C, P * F)
        if pad > 0:
            x = x[..., :x.shape[-1] - pad]
        if x.shape[-1] != T:
            x = x[..., :T]
        return x

    def forward(self, x):
        T = x.shape[-1]
        # Грубый поиск top-k частот через средний модуль спектра
        amp = torch.fft.rfft(x[:, 0, :], dim=-1).abs().mean(0)[1:]
        k = int(min(self.topk, amp.numel()))
        if k <= 0:
            P_list = [T]
            weights = torch.tensor([1.0], device=x.device, dtype=x.dtype)
        else:
            vals, idx = torch.topk(amp, k)
            freqs = (idx + 1)
            P_list = [max(1, int(round(T / int(f.item())))) for f in freqs]
            weights = torch.softmax(vals.to(x.device, dtype=x.dtype), dim=0)

        outs = []
        for P in P_list:
            x2d, pad, T0 = self._reshape_2d(x, P)
            y2d = self.msunit(x2d)
            y1d = self._reshape_back(y2d, pad, T0)
            outs.append(y1d)

        y = torch.zeros_like(x)
        for w, yi in zip(weights, outs):
            y = y + w * yi

        y_flat = y.squeeze(1)          # (B, T)
        out = self.head(y_flat)        # (B, H)
        return out


class TimesNetV2(nn.Module):
    def __init__(self, lookback, horizon, d_model=32, layers=2, topk=2):
        super().__init__()
        self.blocks = nn.ModuleList([
            TimesBlockV2(lookback, horizon, d_model, topk)
            for _ in range(layers)
        ])
        self.horizon = horizon

    def forward(self, x):
        out = None
        y = x
        for blk in self.blocks:
            f = blk(y)                 # (B,H)
            out = f if out is None else (out + f)
        # якорим уровень на последнюю точку обучающего окна
        anchor = x[:, :, -1].squeeze(1).unsqueeze(1)  # (B,1)
        out = out + anchor
        return out
