# models/nbeats_v2.py
import torch
import torch.nn as nn

class NBEATBlock(nn.Module):
    def __init__(self, backcast, forecast, width=128, depth=2):
        super().__init__(); layers=[]; n=backcast
        for _ in range(depth): layers += [nn.Linear(n, width), nn.ReLU()]; n=width
        self.fc = nn.Sequential(*layers); self.theta_b = nn.Linear(width, backcast); self.theta_f = nn.Linear(width, forecast)
    def forward(self, x):
        h = self.fc(x); return self.theta_b(h), self.theta_f(h)

class NBEATSV2(nn.Module):
    def __init__(self, lookback, horizon, width=128, depth=2, nblocks=2):
        super().__init__(); self.blocks=nn.ModuleList([NBEATBlock(lookback,horizon,width,depth) for _ in range(nblocks)]); self.horizon=horizon
    def forward(self, x):
        B,C,T = x.shape; backcast = x.view(B,T); forecast = torch.zeros(B, self.horizon, device=x.device, dtype=x.dtype)
        for blk in self.blocks:
            b,f = blk(backcast); backcast = backcast - b; forecast = forecast + f
        anchor = x[:, :, -1].squeeze(1).unsqueeze(1)   # (B,1)
        forecast = forecast + anchor                              # якорим уровень на последнюю точку
        return forecast