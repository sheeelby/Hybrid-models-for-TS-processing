"""Hybrid MODWT + neural base model forecaster."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch

from ..training import TrainConfig, WindowDatasetStd, train_model
from . import modwt as ext_modwt


def _filter_len(wname: str) -> int:
    try:
        import pywt

        return len(pywt.Wavelet(wname).dec_lo)
    except Exception:
        return 8


def _jmax_for_len(n: int, L: int) -> int:
    if n <= 2:
        return 1
    J = 1
    while (2 ** (J - 1)) * (L - 1) < n:
        J += 1
    return max(1, J - 1)


def modwt_decompose(y, wavelet: str = "db4", level: int = 1, check: bool = True):
    y = np.asarray(y, float)
    N = y.size
    if N < 2:
        return y, []
    L = _filter_len(wavelet)
    Jmax = _jmax_for_len(N, L)
    J = int(max(1, min(level, Jmax)))
    W = ext_modwt.modwt(y, wavelet, J)
    M = ext_modwt.modwtmra(W, wavelet)
    arr = (
        np.vstack([np.asarray(c, float) for c in M])
        if isinstance(M, (list, tuple))
        else np.asarray(M, float)
    )
    if check and (arr.ndim != 2 or arr.shape != (J + 1, N)):
        raise ValueError(f"modwtmra shape={arr.shape}, expected {(J + 1, N)}")
    A = arr[-1]
    D = [arr[j] for j in range(J - 1, -1, -1)]
    return A, D


@dataclass
class HybridComponent:
    model: Optional[torch.nn.Module]
    mu: float
    sd: float
    lookback: Optional[int]


class HybridPlus:
    def __init__(self, base_model_fn, cfg: TrainConfig, wavelet: str = "db4", level: int = 1):
        self.base_model_fn = base_model_fn
        self.cfg = cfg
        self.wavelet = wavelet
        self.level = level
        self.components: List[HybridComponent] = []

    def _prepare_component(self, comp: np.ndarray) -> HybridComponent:
        Lmax = len(comp) - self.cfg.horizon
        if Lmax < 16:
            return HybridComponent(None, float(comp.mean()), float(comp.std() + 1e-8), None)
        L = min(self.cfg.lookback, Lmax)
        ds = WindowDatasetStd(comp, L, self.cfg.horizon, stride=1, scale=True)
        mu, sd = ds.scaler
        model = self.base_model_fn(self.cfg)
        trained = train_model(model, ds, self.cfg)
        return HybridComponent(trained, mu, sd, L)

    def fit(self, y):
        y = np.asarray(y, float)
        A, D = modwt_decompose(y, wavelet=self.wavelet, level=self.level, check=True)
        comps = [A] + D if len(D) else [A]
        self.components = [self._prepare_component(comp) for comp in comps]
        return self

    def forecast(self, y):
        y = np.asarray(y, float)
        H = self.cfg.horizon
        A, D = modwt_decompose(y, wavelet=self.wavelet, level=self.level, check=True)
        comps = [A] + D if len(D) else [A]
        preds = []
        for component, comp in zip(self.components, comps):
            if component.model is None or component.lookback is None:
                continue
            xb = ((comp[-component.lookback :] - component.mu) / component.sd).astype(np.float32)
            xb = xb.reshape(1, 1, -1)
            with torch.no_grad():
                forecast = (
                    component.model(torch.from_numpy(xb).to(self.cfg.device))
                    .cpu()
                    .numpy()
                    .ravel()
                )
            preds.append(forecast * component.sd + component.mu)

        if not preds:
            return np.repeat(y[-1], H).astype(float)
        yhat = np.sum(np.stack(preds, 0), axis=0)
        if yhat.size > H:
            return yhat[:H]
        if yhat.size < H:
            return np.pad(yhat, (0, H - yhat.size), mode="edge")
        return yhat


__all__ = ["HybridPlus", "modwt_decompose"]
