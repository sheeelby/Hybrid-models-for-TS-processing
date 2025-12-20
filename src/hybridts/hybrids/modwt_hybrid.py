from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import TensorDataset

from ..training import TrainConfig, WindowDatasetStd, train_model
from ..models.classic import auto_arima_forecast, ets_forecast
from . import modwt as ext_modw


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
    return modwt_decompose_with_boundary(y, wavelet=wavelet, level=level, boundary="wrap", check=check)


def modwt_decompose_with_boundary(
    y,
    *,
    wavelet: str = "db4",
    level: int = 1,
    boundary: str = "wrap",
    check: bool = True,
):
    y = np.asarray(y, float)
    N = y.size
    if N < 2:
        return y, []
    L = _filter_len(wavelet)
    Jmax = _jmax_for_len(N, L)
    J = int(max(1, min(level, Jmax)))

    boundary = str(boundary).lower()
    if boundary not in {"wrap", "reflect", "mirror", "nearest"}:
        raise ValueError(f"Unsupported boundary='{boundary}' (expected wrap/reflect/mirror/nearest)")

    def _pad(x: np.ndarray, pad_len: int) -> np.ndarray:
        if pad_len <= 0:
            return x
        if x.size < 2:
            return np.pad(x, (pad_len, pad_len), mode="edge")
        if boundary == "nearest":
            left = np.repeat(x[0], pad_len)
            right = np.repeat(x[-1], pad_len)
            return np.concatenate([left, x, right])
        if boundary == "mirror":
            left = x[:pad_len][::-1]
            right = x[-pad_len:][::-1]
            return np.concatenate([left, x, right])
        left = x[1 : pad_len + 1][::-1] if x.size > 1 else np.repeat(x[0], pad_len)
        right = x[-pad_len - 1 : -1][::-1] if x.size > 1 else np.repeat(x[-1], pad_len)
        if left.size < pad_len:
            left = np.pad(left, (pad_len - left.size, 0), mode="edge")
        if right.size < pad_len:
            right = np.pad(right, (0, pad_len - right.size), mode="edge")
        return np.concatenate([left, x, right])

    pad_len = 0
    y_work = y

    if boundary != "wrap" and N >= 3:
        pad_len = int(min(max(8, (2 ** J) * L), max(1, N - 1)))
        y_work = _pad(y, pad_len)

    W = ext_modwt.modwt(y_work, wavelet, J, mode="wrap")
    M = ext_modwt.modwtmra(W, wavelet, mode="wrap")
    arr = (
        np.vstack([np.asarray(c, float) for c in M])
        if isinstance(M, (list, tuple))
        else np.asarray(M, float)
    )
    if pad_len > 0:
        arr = arr[:, pad_len : pad_len + N]
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
    per_series_scaling: bool = False


def build_global_hybrid_components(
    pairs: Sequence[Tuple[str, np.ndarray, np.ndarray]],
    cfg: TrainConfig,
    base_model_fn,
    wavelet: str = "db4",
    level: int = 1,
    boundary: str = "wrap",
) -> List[HybridComponent]:

    if not pairs:
        return []

    comp_values: List[List[np.ndarray]] = []
    for _, y_tr, _ in pairs:
        y = np.asarray(y_tr, float)
        if y.size < 2:
            continue
        A, D = modwt_decompose_with_boundary(
            y, wavelet=wavelet, level=level, boundary=boundary, check=True
        )
        comps = [A] + D if len(D) else [A]
        if not comp_values:
            comp_values = [[] for _ in range(len(comps))]
        for idx, comp in enumerate(comps):
            if idx >= len(comp_values):
                comp_values.append([])
            comp_values[idx].append(np.asarray(comp, float))

    components: List[HybridComponent] = []
    H = cfg.horizon
    L = cfg.lookback
    for comp_series in comp_values:
        if not comp_series:
            components.append(HybridComponent(None, 0.0, 1.0, None, per_series_scaling=True))
            continue
        X_list: List[np.ndarray] = []
        Y_list: List[np.ndarray] = []
        for arr in comp_series:
            arr_f = arr.astype(np.float32)
            mu = float(arr_f.mean())
            sd = float(arr_f.std() + 1e-8)
            z = (arr_f - mu) / sd
            if z.size < L + H:
                continue
            for t in range(0, len(z) - L - H + 1):
                X_list.append(z[t : t + L])
                Y_list.append(z[t + L : t + L + H])
        if not X_list:
            components.append(HybridComponent(None, 0.0, 1.0, None, per_series_scaling=True))
            continue
        X = np.stack(X_list, axis=0).astype(np.float32)
        Y = np.stack(Y_list, axis=0).astype(np.float32)
        ds = TensorDataset(
            torch.from_numpy(X).unsqueeze(1),
            torch.from_numpy(Y),
        )
        model = base_model_fn(cfg)
        trained = train_model(model, ds, cfg)
        components.append(HybridComponent(trained, 0.0, 1.0, L, per_series_scaling=True))
    return components


class HybridPlus:
    def __init__(
        self,
        base_model_fn,
        cfg: TrainConfig,
        wavelet: str = "db4",
        level: int = 1,
        boundary: str = "wrap",
        pretrained_components: Optional[List[HybridComponent]] = None,
        seasonal_period: int | None = None,
    ):
        self.base_model_fn = base_model_fn
        self.cfg = cfg
        self.wavelet = wavelet
        self.level = level
        self.boundary = str(boundary).lower()
        self.components: List[HybridComponent] = []
        self.pretrained_components = pretrained_components
        self.seasonal_period = int(seasonal_period) if seasonal_period is not None else None

    def _prepare_component(self, comp: np.ndarray) -> HybridComponent:
        Lmax = len(comp) - self.cfg.horizon
        if Lmax < 16:
            return HybridComponent(None, float(comp.mean()), float(comp.std() + 1e-8), None, per_series_scaling=False)
        L = min(self.cfg.lookback, Lmax)
        ds = WindowDatasetStd(comp, L, self.cfg.horizon, stride=1, scale=True)
        mu, sd = ds.scaler
        model = self.base_model_fn(self.cfg)
        trained = train_model(model, ds, self.cfg)
        return HybridComponent(trained, mu, sd, L, per_series_scaling=False)

    def fit(self, y, *, components_override: Optional[List[np.ndarray]] = None):
        y = np.asarray(y, float)
        if components_override is None:
            A, D = modwt_decompose_with_boundary(
                y, wavelet=self.wavelet, level=self.level, boundary=self.boundary, check=True
            )
            comps = [A] + D if len(D) else [A]
        else:
            comps = [np.asarray(c, float) for c in components_override]
            if not comps:
                comps = [y]
        if self.pretrained_components is not None and len(self.pretrained_components) == len(comps):
            # Reuse globally trained components; no per-series training.
            self.components = self.pretrained_components
        else:
            self.components = [self._prepare_component(comp) for comp in comps]
        return self

    def forecast(self, y, *, components_override: Optional[List[np.ndarray]] = None):
        y = np.asarray(y, float)
        H = self.cfg.horizon
        if components_override is None:
            A, D = modwt_decompose_with_boundary(
                y, wavelet=self.wavelet, level=self.level, boundary=self.boundary, check=True
            )
            comps = [A] + D if len(D) else [A]
        else:
            comps = [np.asarray(c, float) for c in components_override]
            if not comps:
                comps = [y]
            A = comps[0]
        if not self.components:
            return np.repeat(y[-1], H).astype(float)

        comp_preds: List[np.ndarray] = []
        for component, comp in zip(self.components, comps):
            if component.model is None or component.lookback is None:
                comp_preds.append(np.repeat(float(comp[-1]) if comp.size else 0.0, H).astype(float))
                continue
            if component.per_series_scaling:
                mu = float(np.mean(comp))
                sd = float(np.std(comp) + 1e-8)
            else:
                mu = float(component.mu)
                sd = float(component.sd)
            xb = ((comp[-component.lookback :] - mu) / sd).astype(np.float32)
            xb = xb.reshape(1, 1, -1)
            with torch.no_grad():
                forecast = component.model(torch.from_numpy(xb).to(self.cfg.device)).cpu().numpy().ravel()
            pred = forecast * sd + mu
            if pred.size > H:
                pred = pred[:H]
            if pred.size < H:
                pred = np.pad(pred, (0, H - pred.size), mode="edge")
            comp_preds.append(pred.astype(float, copy=False))

        if len(comp_preds) >= 2 and A.size > 0 and y.size > 0:
            a_last = float(A[-1])
            resid_last = float(y[-1] - a_last)
            resid_pred = np.sum(np.stack(comp_preds[1:], 0), axis=0)
            k = int(max(1, min(6, H)))
            if k == 1:
                resid_pred = resid_pred.astype(float, copy=True)
                resid_pred[0] = 0.6 * resid_last + 0.4 * float(resid_pred[0])
            else:
                resid_pred = resid_pred.astype(float, copy=True)
                alpha0 = 0.6
                for i in range(k):
                    alpha = float(alpha0 * (k - 1 - i) / (k - 1))
                    resid_pred[i] = alpha * resid_last + (1.0 - alpha) * resid_pred[i]
            yhat = comp_preds[0] + resid_pred
        else:
            yhat = np.sum(np.stack(comp_preds, 0), axis=0)
        if yhat.size > H:
            yhat = yhat[:H]
        if yhat.size < H:
            yhat = np.pad(yhat, (0, H - yhat.size), mode="edge")
        return _stabilize_reconstruction_boundary(y, yhat)

    def forecast_components(self, y, *, components_override: Optional[List[np.ndarray]] = None) -> dict[str, np.ndarray]:
        """Return per-component forecasts used for reconstruction.

        Keys: "A_J", "D_1", ..., "D_J".
        """
        y = np.asarray(y, float)
        H = self.cfg.horizon
        if components_override is None:
            A, D = modwt_decompose_with_boundary(
                y, wavelet=self.wavelet, level=self.level, boundary=self.boundary, check=True
            )
            comps = [A] + D if len(D) else [A]
        else:
            comps = [np.asarray(c, float) for c in components_override]
            if not comps:
                comps = [y]
            A = comps[0]
        if not self.components:
            return {"A_J": np.repeat(float(A[-1]) if A.size else 0.0, H).astype(float)}

        comp_preds: List[np.ndarray] = []
        for component, comp in zip(self.components, comps):
            if component.model is None or component.lookback is None:
                comp_preds.append(np.repeat(float(comp[-1]) if comp.size else 0.0, H).astype(float))
                continue
            if component.per_series_scaling:
                mu = float(np.mean(comp))
                sd = float(np.std(comp) + 1e-8)
            else:
                mu = float(component.mu)
                sd = float(component.sd)
            xb = ((comp[-component.lookback :] - mu) / sd).astype(np.float32)
            xb = xb.reshape(1, 1, -1)
            with torch.no_grad():
                forecast = component.model(torch.from_numpy(xb).to(self.cfg.device)).cpu().numpy().ravel()
            pred = forecast * sd + mu
            if pred.size > H:
                pred = pred[:H]
            if pred.size < H:
                pred = np.pad(pred, (0, H - pred.size), mode="edge")
            comp_preds.append(pred.astype(float, copy=False))

        if len(comp_preds) >= 2 and A.size > 0 and y.size > 0:
            a_last = float(A[-1])
            resid_last = float(y[-1] - a_last)
            resid_pred = np.sum(np.stack(comp_preds[1:], 0), axis=0).astype(float, copy=True)
            k = int(max(1, min(6, H)))
            if k == 1:
                resid_pred[0] = 0.6 * resid_last + 0.4 * float(resid_pred[0])
            else:
                alpha0 = 0.6
                for i in range(k):
                    alpha = float(alpha0 * (k - 1 - i) / (k - 1))
                    resid_pred[i] = alpha * resid_last + (1.0 - alpha) * resid_pred[i]

            raw = np.sum(np.stack(comp_preds[1:], 0), axis=0)
            diff = resid_pred - raw
            details = [p.astype(float, copy=True) for p in comp_preds[1:]]
            weights = np.stack([np.abs(p) for p in details], 0)
            denom = np.sum(weights, axis=0)
            if np.all(denom < 1e-8):
                for j in range(len(details)):
                    details[j] = details[j] + diff / float(len(details))
            else:
                for j in range(len(details)):
                    details[j] = details[j] + diff * (weights[j] / (denom + 1e-8))
            comp_preds = [comp_preds[0]] + details

        out: dict[str, np.ndarray] = {"A_J": np.asarray(comp_preds[0], float)}
        for j, pred in enumerate(comp_preds[1:], start=1):
            out[f"D_{j}"] = np.asarray(pred, float)
        return out


class VWHybridMixed:
    def __init__(
        self,
        *,
        aj_model_fn,
        detail_method: str,
        cfg: TrainConfig,
        wavelet: str = "db4",
        level: int = 1,
        seasonal_period: int | None = None,
        boundary: str = "reflect",
        detail_anchor: bool = True,
        detail_transition_steps: int = 3,
        detail_dampen: bool = True,
        detail_dampen_tau: float = 3.0,
        residual_anchor: bool = True,
        residual_transition_steps: int = 6,
    ) -> None:
        self.aj_model_fn = aj_model_fn
        self.detail_method = str(detail_method).lower()
        self.cfg = cfg
        self.wavelet = wavelet
        self.level = int(level)
        self.seasonal_period = int(seasonal_period) if seasonal_period is not None else None
        self.boundary = str(boundary).lower()
        self.detail_anchor = bool(detail_anchor)
        self.detail_transition_steps = int(detail_transition_steps)
        self.detail_dampen = bool(detail_dampen)
        self.detail_dampen_tau = float(detail_dampen_tau)
        self.residual_anchor = bool(residual_anchor)
        self.residual_transition_steps = int(residual_transition_steps)
        self.aj_component: HybridComponent | None = None

    def _prepare_component(self, comp: np.ndarray) -> HybridComponent:
        Lmax = len(comp) - self.cfg.horizon
        if Lmax < 16:
            return HybridComponent(None, float(comp.mean()), float(comp.std() + 1e-8), None)
        L = min(self.cfg.lookback, Lmax)
        ds = WindowDatasetStd(comp, L, self.cfg.horizon, stride=1, scale=True)
        mu, sd = ds.scaler
        model = self.aj_model_fn(self.cfg)
        trained = train_model(model, ds, self.cfg)
        return HybridComponent(trained, mu, sd, L)

    def fit(self, y, *, components_override: Optional[List[np.ndarray]] = None):
        y = np.asarray(y, float)
        if components_override is None:
            A, _ = modwt_decompose_with_boundary(
                y, wavelet=self.wavelet, level=self.level, boundary=self.boundary, check=True
            )
        else:
            comps = [np.asarray(c, float) for c in components_override]
            A = comps[0] if comps else y
        self.aj_component = self._prepare_component(A)
        return self

    def _forecast_neural(self, component: HybridComponent, comp: np.ndarray) -> np.ndarray:
        H = self.cfg.horizon
        if component.model is None or component.lookback is None:
            return np.repeat(float(comp[-1]) if comp.size else 0.0, H).astype(float)
        xb = ((comp[-component.lookback :] - component.mu) / component.sd).astype(np.float32)
        xb = xb.reshape(1, 1, -1)
        with torch.no_grad():
            forecast = (
                component.model(torch.from_numpy(xb).to(self.cfg.device)).cpu().numpy().ravel()
            )
        pred = forecast * component.sd + component.mu
        if pred.size > H:
            pred = pred[:H]
        if pred.size < H:
            pred = np.pad(pred, (0, H - pred.size), mode="edge")
        return pred.astype(float, copy=False)

    def _forecast_detail(self, comp: np.ndarray) -> np.ndarray:
        H = self.cfg.horizon
        comp = np.asarray(comp, float)
        if comp.size == 0:
            return np.zeros(H, dtype=float)

        if self.detail_method in {"ets", "holtwinters"}:
            seasonal_periods = self.seasonal_period if self.seasonal_period and self.seasonal_period > 1 else None
            seasonal = "add" if seasonal_periods else None
            pred = ets_forecast(comp, H, seasonal_periods=seasonal_periods, trend=None, seasonal=seasonal)
        elif self.detail_method in {"arima_auto", "auto_arima"}:
            pred = auto_arima_forecast(comp, H)
        else:
            raise ValueError(
                f"Unknown detail_method='{self.detail_method}' (expected 'ets' or 'arima_auto')"
            )

        pred = np.asarray(pred, float)
        if pred.size > H:
            pred = pred[:H]
        if pred.size < H:
            pred = np.pad(pred, (0, H - pred.size), mode="edge")

        out = pred.astype(float, copy=True)

        if self.detail_dampen and H > 0:
            tau = max(0.5, float(self.detail_dampen_tau))
            w = np.exp(-np.arange(H, dtype=float) / tau)
            out = out * w

        if not self.detail_anchor or comp.size < 2 or H <= 0:
            return out.astype(float, copy=False)

        last = float(comp[-1])
        k = int(max(1, min(self.detail_transition_steps, H)))
        for i in range(k):
            alpha = float((k - 1 - i) / max(1, k - 1)) if k > 1 else 1.0
            out[i] = alpha * last + (1.0 - alpha) * out[i]
        return out

    def forecast(self, y, *, components_override: Optional[List[np.ndarray]] = None):
        y = np.asarray(y, float)
        H = self.cfg.horizon
        if components_override is None:
            A, D = modwt_decompose_with_boundary(
                y, wavelet=self.wavelet, level=self.level, boundary=self.boundary, check=True
            )
        else:
            comps = [np.asarray(c, float) for c in components_override]
            A = comps[0] if comps else y
            D = comps[1:] if len(comps) > 1 else []
        if self.aj_component is None:
            raise RuntimeError("Call fit() before forecast()")

        aj_pred = self._forecast_neural(self.aj_component, A)
        detail_preds: List[np.ndarray] = []
        for dj in D:
            pred = self._forecast_detail(np.asarray(dj, float))
            if pred.size > H:
                pred = pred[:H]
            if pred.size < H:
                pred = np.pad(pred, (0, H - pred.size), mode="edge")
            detail_preds.append(pred.astype(float, copy=False))

        if detail_preds:
            resid_pred = np.sum(np.stack(detail_preds, 0), axis=0)
        else:
            resid_pred = np.zeros(H, dtype=float)

        if self.residual_anchor and y.size > 0 and A.size > 0 and H > 0:
            resid_last = float(y[-1] - A[-1])
            k = int(max(1, min(self.residual_transition_steps, H)))
            if k == 1:
                resid_pred = resid_pred.astype(float, copy=True)
                resid_pred[0] = 0.6 * resid_last + 0.4 * float(resid_pred[0])
            else:
                resid_pred = resid_pred.astype(float, copy=True)
                alpha0 = 0.6
                for i in range(k):
                    alpha = float(alpha0 * (k - 1 - i) / (k - 1))
                    resid_pred[i] = alpha * resid_last + (1.0 - alpha) * resid_pred[i]

        yhat = aj_pred + resid_pred
        if yhat.size > H:
            yhat = yhat[:H]
        if yhat.size < H:
            yhat = np.pad(yhat, (0, H - yhat.size), mode="edge")
        return _stabilize_reconstruction_boundary(y, yhat)

    def forecast_components(self, y, *, components_override: Optional[List[np.ndarray]] = None) -> dict[str, np.ndarray]:

        y = np.asarray(y, float)
        H = self.cfg.horizon
        if components_override is None:
            A, D = modwt_decompose_with_boundary(
                y, wavelet=self.wavelet, level=self.level, boundary=self.boundary, check=True
            )
        else:
            comps = [np.asarray(c, float) for c in components_override]
            A = comps[0] if comps else y
            D = comps[1:] if len(comps) > 1 else []
        if self.aj_component is None:
            raise RuntimeError("Call fit() before forecast_components()")

        aj_pred = self._forecast_neural(self.aj_component, A)
        detail_preds: List[np.ndarray] = []
        for dj in D:
            pred = self._forecast_detail(np.asarray(dj, float))
            if pred.size > H:
                pred = pred[:H]
            if pred.size < H:
                pred = np.pad(pred, (0, H - pred.size), mode="edge")
            detail_preds.append(pred.astype(float, copy=False))

        if detail_preds:
            resid_pred = np.sum(np.stack(detail_preds, 0), axis=0).astype(float, copy=True)
        else:
            resid_pred = np.zeros(H, dtype=float)

        if self.residual_anchor and y.size > 0 and A.size > 0 and H > 0:
            resid_last = float(y[-1] - A[-1])
            k = int(max(1, min(self.residual_transition_steps, H)))
            if k == 1:
                resid_pred[0] = 0.6 * resid_last + 0.4 * float(resid_pred[0])
            else:
                alpha0 = 0.6
                for i in range(k):
                    alpha = float(alpha0 * (k - 1 - i) / (k - 1))
                    resid_pred[i] = alpha * resid_last + (1.0 - alpha) * resid_pred[i]

        if detail_preds:
            raw = np.sum(np.stack(detail_preds, 0), axis=0)
            diff = resid_pred - raw
            details = [p.astype(float, copy=True) for p in detail_preds]
            weights = np.stack([np.abs(p) for p in details], 0)
            denom = np.sum(weights, axis=0)
            if np.all(denom < 1e-8):
                for j in range(len(details)):
                    details[j] = details[j] + diff / float(len(details))
            else:
                for j in range(len(details)):
                    details[j] = details[j] + diff * (weights[j] / (denom + 1e-8))
        else:
            details = []

        out: dict[str, np.ndarray] = {"A_J": np.asarray(aj_pred, float)}
        for j, pred in enumerate(details, start=1):
            out[f"D_{j}"] = np.asarray(pred, float)
        return out


def _stabilize_reconstruction_boundary(y: np.ndarray, yhat: np.ndarray) -> np.ndarray:
    if yhat.size == 0 or y.size < 3:
        return yhat.astype(float, copy=False)
    k = int(min(24, y.size - 1))
    diffs = y[-k:] - y[-k - 1 : -1]
    abs_diffs = np.abs(diffs)
    scale = float(np.quantile(abs_diffs, 0.75) + 1e-8)
    max_jump = 4.0 * scale
    jump0 = float(yhat[0] - y[-1])
    last_diff = float(y[-1] - y[-2])
    bad_direction = (last_diff * jump0 < 0) and (abs(last_diff) > 0.5 * scale) and (abs(jump0) > 0.5 * scale)
    yhat2 = yhat.astype(float, copy=True)

    if abs(jump0) > max_jump or bad_direction:
        if bad_direction and abs(jump0) <= max_jump:
            jump0 = float(np.sign(last_diff) * abs(jump0))
        yhat2[0] = float(y[-1]) + float(np.clip(jump0, -max_jump, max_jump))

    if yhat2.size >= 3:
        pred_diffs = np.diff(yhat2[: min(8, yhat2.size)])
        pred_scale = float(np.quantile(np.abs(pred_diffs), 0.75) + 1e-8) if pred_diffs.size else scale
        spike_scale = float(max(scale, pred_scale))
        d01 = float(yhat2[0] - yhat2[1])
        d12 = float(yhat2[1] - yhat2[2])
        if (abs(d01) > 6.0 * spike_scale) and (abs(d12) < 2.5 * spike_scale) and (abs(yhat2[0] - y[-1]) > 4.0 * spike_scale):
            target = float(yhat2[1])
            yhat2[0] = float(y[-1]) + float(np.clip(target - float(y[-1]), -max_jump, max_jump))

    return yhat2


__all__ = [
    "HybridComponent",
    "HybridPlus",
    "VWHybridMixed",
    "build_global_hybrid_components",
    "modwt_decompose",
    "modwt_decompose_with_boundary",
]
