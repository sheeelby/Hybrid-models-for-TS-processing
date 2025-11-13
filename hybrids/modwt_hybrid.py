# hybrids/modwt_hybrid.py
import numpy as np, torch
from pathlib import Path
from importlib.util import spec_from_file_location, module_from_spec

from train_utils import WindowDatasetStd, TrainConfig, train_model

_THIS_DIR = Path(__file__).resolve().parent
EXT_MODWT = _THIS_DIR / "modwt.py"
_spec = spec_from_file_location("ext_modwt", EXT_MODWT)
ext_modwt = module_from_spec(_spec); _spec.loader.exec_module(ext_modwt)  # type: ignore

def _filter_len(wname: str):
    try:
        import pywt
        return len(pywt.Wavelet(wname).dec_lo)
    except Exception:
        return 8

def _jmax_for_len(n:int, L:int) -> int:
    if n <= 2: return 1
    J = 1
    while (2**(J-1))*(L-1) < n:
        J += 1
    return max(1, J-1)

def modwt_decompose(y, wavelet='db4', level=1, check=True):
    y = np.asarray(y, float); N = y.size
    if N < 2: return y, []
    L = _filter_len(wavelet); Jmax = _jmax_for_len(N, L); J = int(max(1, min(level, Jmax)))
    W = ext_modwt.modwt(y, wavelet, J)
    M = ext_modwt.modwtmra(W, wavelet)  # [D1..DJ, A_J] или (J+1,N)
    arr = np.vstack([np.asarray(c, float) for c in M]) if isinstance(M, (list, tuple)) else np.asarray(M, float)
    if check and (arr.ndim != 2 or arr.shape != (J+1, N)):
        raise ValueError(f"modwtmra shape={arr.shape}, expected {(J+1, N)}")
    A = arr[-1]; D = [arr[j] for j in range(J-1, -1, -1)]
    return A, D

class HybridPlus:
    def __init__(self, base_model_fn, cfg: TrainConfig, wavelet='db4', level=1):
        self.base_model_fn = base_model_fn
        self.cfg = cfg
        self.wavelet = wavelet
        self.level = level
        self.models = []
        self.lookbacks = []

    def fit(self, y):
        y = np.asarray(y, float)
        A, D = modwt_decompose(y, wavelet=self.wavelet, level=self.level, check=True)
        comps = [A] + D if len(D) else [A]
        self.models, self.lookbacks = [], []
        for comp in comps:
            Lmax = len(comp) - self.cfg.horizon
            if Lmax < 16:
                self.models.append(None); self.lookbacks.append(None); continue
            L = min(self.cfg.lookback, Lmax)
            ds = WindowDatasetStd(comp, L, self.cfg.horizon, stride=1, scale=True)
            mu, sd = ds.scaler
            m = self.base_model_fn(self.cfg)
            m = train_model(m, ds, self.cfg)
            self.models.append((m, mu, sd)); self.lookbacks.append(L)
        return self

    def forecast(self, y):
        y = np.asarray(y, float); H = self.cfg.horizon
        A, D = modwt_decompose(y, wavelet=self.wavelet, level=self.level, check=True)
        comps = [A] + D if len(D) else [A]
        preds=[]
        for info, Lc, comp in zip(self.models, self.lookbacks, comps):
            if info is None or Lc is None: continue
            m, mu, sd = info
            xb = ((comp[-Lc:] - mu)/sd).astype(np.float32).reshape(1,1,-1)
            with torch.no_grad():
                p = m(torch.from_numpy(xb).to(self.cfg.device)).cpu().numpy().ravel()
            preds.append(p*sd + mu)
        if not preds: return np.repeat(y[-1], H).astype(float)
        yhat = np.sum(np.stack(preds, 0), axis=0)
        return yhat[:H] if yhat.size>H else (np.pad(yhat, (0, H-yhat.size), mode='edge') if yhat.size<H else yhat)
