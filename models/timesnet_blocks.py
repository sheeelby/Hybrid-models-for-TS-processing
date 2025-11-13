# models/timesnet_blocks.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class MSUnit(nn.Module):
    def __init__(self, in_ch=1, d_model=32):
        super().__init__()
        self.act = nn.GELU()
        # без встроенного padding — паддим вручную для причинности
        self.p3 = nn.Conv2d(in_ch, d_model, kernel_size=(3,1), padding=0, bias=False)
        self.p5 = nn.Conv2d(in_ch, d_model, kernel_size=(5,1), padding=0, bias=False)
        self.f3 = nn.Conv2d(in_ch, d_model, kernel_size=(1,3), padding=0, bias=False)
        self.f5 = nn.Conv2d(in_ch, d_model, kernel_size=(1,5), padding=0, bias=False)

        self.proj = nn.Conv2d(4*d_model, in_ch, kernel_size=1, bias=False)
        self.norm = nn.GroupNorm(1, 4*d_model)

    def _causal_T(self, conv, x):
        kT = conv.kernel_size[0]
        # паддинг слева по времени (H): pad=(Wleft,Wright,Htop,Hbottom)
        x_pad = F.pad(x, (0, 0, kT-1, 0))
        return conv(x_pad)

    def _causal_F(self, conv, x):
        kF = conv.kernel_size[1]
        # паддинг слева по «частотному» измерению (W)
        x_pad = F.pad(x, (kF-1, 0, 0, 0))
        return conv(x_pad)

    def forward(self, x):
        a = self.act(self._causal_T(self.p3, x))
        b = self.act(self._causal_T(self.p5, x))
        c = self.act(self._causal_F(self.f3, x))
        d = self.act(self._causal_F(self.f5, x))
        y = torch.cat([a, b, c, d], dim=1)
        y = self.norm(y)
        y = self.proj(y)
        return y
