from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import TensorDataset

from ..training import TrainConfig, WindowDatasetStd, train_model
from ..models.classic import auto_arima_forecast, ets_forecast
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
    blend_weight: float = 1.0


@dataclass(frozen=True)
class VWDetailPolicy:
    method: str = "zero"
    seasonal_period: int | None = None
    blend_zero: float = 1.0
    amp_limit: float | None = None


def _persistence_forecast(comp: np.ndarray, H: int) -> np.ndarray:
    if H <= 0:
        return np.zeros(0, dtype=float)
    if comp.size == 0:
        return np.zeros(H, dtype=float)
    return np.repeat(float(comp[-1]), H).astype(float)


def _neural_component_pred(component: HybridComponent, comp: np.ndarray, H: int, device: str) -> np.ndarray:
    comp = np.asarray(comp, float)
    if component.model is None or component.lookback is None:
        return _persistence_forecast(comp, H)
    if comp.size < component.lookback:
        return _persistence_forecast(comp, H)

    if component.per_series_scaling:
        mu = float(np.mean(comp))
        sd = float(np.std(comp) + 1e-8)
    else:
        mu = float(component.mu)
        sd = float(component.sd)
    xb = ((comp[-component.lookback :] - mu) / sd).astype(np.float32)
    xb = xb.reshape(1, 1, -1)
    with torch.no_grad():
        forecast = component.model(torch.from_numpy(xb).to(device)).cpu().numpy().ravel()
    pred = forecast * sd + mu
    if pred.size > H:
        pred = pred[:H]
    if pred.size < H:
        pred = np.pad(pred, (0, H - pred.size), mode="edge")
    return pred.astype(float, copy=False)


def _blend_with_persistence(
    pred: np.ndarray,
    comp: np.ndarray,
    H: int,
    blend_weight: float,
) -> np.ndarray:
    w = float(np.clip(blend_weight, 0.0, 1.0))
    if w >= 0.999:
        return np.asarray(pred, float)
    base = _persistence_forecast(np.asarray(comp, float), H)
    return (w * np.asarray(pred, float) + (1.0 - w) * base).astype(float, copy=False)


def _weighted_mse(pred: np.ndarray, target: np.ndarray, weights: np.ndarray) -> float:
    err = np.asarray(pred, float) - np.asarray(target, float)
    return float(np.mean(weights * (err**2)))


def _weighted_rmse(pred: np.ndarray, target: np.ndarray, weights: np.ndarray) -> float:
    return float(np.sqrt(max(0.0, _weighted_mse(pred, target, weights))))


def _component_lookback(
    base_lookback: int,
    *,
    horizon: int,
    Lmax: int,
    component_index: int,
    n_components: int,
) -> int:
    base = int(min(max(1, base_lookback), max(1, Lmax)))
    if Lmax <= 0:
        return 0

    target = base
    if component_index == 0:
        target = max(base, int(round(1.5 * base)), 2 * int(max(1, horizon)))
    elif component_index == 1 and n_components > 2:
        target = max(base, int(round(1.15 * base)))
    return int(max(1, min(int(Lmax), int(target))))


def _normalize_horizon_forecast(pred: np.ndarray, H: int) -> np.ndarray:
    pred = np.asarray(pred, float).ravel()
    if H <= 0:
        return np.zeros(0, dtype=float)
    if pred.size == 0:
        return np.zeros(H, dtype=float)
    if pred.size > H:
        pred = pred[:H]
    if pred.size < H:
        pred = np.pad(pred, (0, H - pred.size), mode="edge")
    return pred.astype(float, copy=False)


def _baseline_forecast_raw(y_hist: np.ndarray, H: int, *, kind: str, period: int | None) -> np.ndarray:
    y_hist = np.asarray(y_hist, float)
    if kind == "last":
        return _persistence_forecast(y_hist, H)
    if kind == "arima_auto":
        return _normalize_horizon_forecast(np.asarray(auto_arima_forecast(y_hist, H), float), H)
    if kind == "ets":
        sp = int(period) if period and int(period) > 1 else None
        seasonal = "add" if sp else None
        pred = ets_forecast(y_hist, H, seasonal_periods=sp, trend="add", seasonal=seasonal)
        return _normalize_horizon_forecast(np.asarray(pred, float), H)
    raise ValueError(f"Unknown raw baseline kind '{kind}'")


def _select_output_blend(
    hybrid_pred: np.ndarray,
    *,
    y_hist: np.ndarray,
    y_tgt: np.ndarray,
    seasonal_periods: Sequence[int] = (),
    min_hybrid_weight: float = 0.65,
    hard_min_hybrid_weight: float = 0.45,
    min_improvement: float = 0.08,
    emergency_ratio: float = 1.4,
) -> tuple[str, float, int | None]:
    y_hist = np.asarray(y_hist, float)
    y_tgt = np.asarray(y_tgt, float)
    H = int(y_tgt.size)
    if H <= 0 or y_hist.size < 8:
        return ("none", 1.0, None)

    hyb = _normalize_horizon_forecast(np.asarray(hybrid_pred, float), H)
    tau = max(1.5, float(min(H, 8)) / 2.0)
    w = np.exp(-np.arange(H, dtype=float) / tau)
    w = w / (np.mean(w) + 1e-12)
    hybrid_score = _weighted_rmse(hyb, y_tgt, w)

    candidate_baselines: list[tuple[str, int | None]] = [("last", None), ("arima_auto", None), ("ets", None)]
    for p in sorted({int(p) for p in seasonal_periods if p is not None and int(p) > 1}):
        if y_hist.size >= 2 * int(p):
            candidate_baselines.append(("ets", int(p)))

    best = (hybrid_score, "none", 1.0, None, hybrid_score)
    for kind, period in candidate_baselines:
        try:
            base = _baseline_forecast_raw(y_hist, H, kind=kind, period=period)
        except Exception:
            continue
        base_score = _weighted_rmse(base, y_tgt, w)

        d = hyb - base
        n = y_tgt - base
        denom = float(np.sum(w * d * d))
        if denom <= 1e-12:
            alpha_opt = 1.0
        else:
            alpha_opt = float(np.clip(np.sum(w * d * n) / denom, 0.0, 1.0))
        for alpha in (alpha_opt, 0.0, 0.25, 0.5, 0.75, 1.0):
            pred = alpha * hyb + (1.0 - alpha) * base
            score = _weighted_rmse(pred, y_tgt, w)
            if score + 1e-12 < best[0]:
                best = (score, kind, float(alpha), period, base_score)

    best_score, kind_best, alpha_best, period_best, base_score_best = best
    if kind_best == "none":
        return ("none", 1.0, None)

    rel_improvement = float((hybrid_score - best_score) / max(hybrid_score, 1e-12))
    if rel_improvement < float(max(0.0, min_improvement)):
        return ("none", 1.0, None)

    alpha_best = float(np.clip(alpha_best, 0.0, 1.0))
    min_hybrid_weight = float(np.clip(min_hybrid_weight, 0.0, 1.0))
    hard_min_hybrid_weight = float(np.clip(hard_min_hybrid_weight, 0.0, min_hybrid_weight))
    emergency_ratio = float(max(1.0, emergency_ratio))

    if alpha_best < min_hybrid_weight:
        allow_hard_fallback = (
            np.isfinite(base_score_best)
            and hybrid_score > float(base_score_best) * emergency_ratio
            and rel_improvement >= max(2.0 * float(min_improvement), 0.12)
        )
        if not allow_hard_fallback:
            try:
                base = _baseline_forecast_raw(y_hist, H, kind=str(kind_best), period=period_best)
            except Exception:
                return ("none", 1.0, None)
            alpha_floor = min_hybrid_weight
            pred_floor = alpha_floor * hyb + (1.0 - alpha_floor) * base
            score_floor = _weighted_rmse(pred_floor, y_tgt, w)
            gain_floor = float((hybrid_score - score_floor) / max(hybrid_score, 1e-12))
            if gain_floor < max(0.5 * float(min_improvement), 0.03):
                return ("none", 1.0, None)
            alpha_best = alpha_floor
        else:
            try:
                base = _baseline_forecast_raw(y_hist, H, kind=str(kind_best), period=period_best)
            except Exception:
                return ("none", 1.0, None)
            alpha_floor = max(hard_min_hybrid_weight, alpha_best)
            pred_floor = alpha_floor * hyb + (1.0 - alpha_floor) * base
            score_floor = _weighted_rmse(pred_floor, y_tgt, w)
            gain_floor = float((hybrid_score - score_floor) / max(hybrid_score, 1e-12))
            if gain_floor < max(0.5 * float(min_improvement), 0.03):
                return ("none", 1.0, None)
            alpha_best = alpha_floor

    if alpha_best >= 0.999:
        return ("none", 1.0, None)
    return (str(kind_best), float(alpha_best), int(period_best) if period_best else None)


def _acf_lag_score(x: np.ndarray, lag: int) -> float:
    x = np.asarray(x, float)
    lag = int(lag)
    if lag <= 1 or x.size <= lag + 2:
        return 0.0
    a = x[:-lag] - float(np.mean(x[:-lag]))
    b = x[lag:] - float(np.mean(x[lag:]))
    denom = float(np.sqrt(np.sum(a * a) * np.sum(b * b)) + 1e-12)
    if denom <= 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def _clip_detail_amplitude(pred: np.ndarray, comp_hist: np.ndarray, amp_limit: float | None) -> np.ndarray:
    if amp_limit is None or not np.isfinite(float(amp_limit)):
        return np.asarray(pred, float)
    lim = float(max(1e-8, amp_limit))
    return np.clip(np.asarray(pred, float), -lim, lim).astype(float, copy=False)


def _calibrate_blend_weight(
    component: HybridComponent,
    comp: np.ndarray,
    *,
    horizon: int,
    device: str,
    max_eval_windows: int = 8,
    min_model_weight: float = 0.55,
    min_improvement: float = 0.03,
    emergency_ratio: float = 1.4,
) -> float:
    comp = np.asarray(comp, float)
    if (
        component.model is None
        or component.lookback is None
        or horizon <= 0
        or comp.size < (component.lookback + horizon + 4)
    ):
        return 1.0

    L = int(component.lookback)
    n_windows = comp.size - L - horizon + 1
    if n_windows <= 0:
        return 1.0

    m = int(max(2, min(max_eval_windows, n_windows)))
    start0 = n_windows - m
    pred_list: list[np.ndarray] = []
    base_list: list[np.ndarray] = []
    tgt_list: list[np.ndarray] = []

    tau = max(1.5, float(min(horizon, 8)) / 2.0)
    h_w = np.exp(-np.arange(horizon, dtype=float) / tau)
    h_w = h_w / (np.mean(h_w) + 1e-12)

    for t in range(start0, n_windows):
        hist = comp[t : t + L]
        tgt = comp[t + L : t + L + horizon]
        if hist.size < L or tgt.size < horizon:
            continue
        try:
            pred = _neural_component_pred(component, hist, horizon, device)
        except Exception:
            return 1.0
        base = _persistence_forecast(hist, horizon)
        pred_list.append(np.asarray(pred, float))
        base_list.append(base)
        tgt_list.append(np.asarray(tgt, float))

    if not pred_list:
        return 1.0

    P = np.concatenate(pred_list, axis=0)
    B = np.concatenate(base_list, axis=0)
    T = np.concatenate(tgt_list, axis=0)
    W = np.tile(h_w, len(pred_list))

    d = P - B
    n = T - B
    denom = float(np.sum(W * d * d))
    if denom <= 1e-12:
        return 1.0
    w_opt = float(np.sum(W * d * n) / denom)
    w_opt = float(np.clip(w_opt, 0.0, 1.0))

    mse_model = _weighted_mse(P, T, W)
    mse_base = _weighted_mse(B, T, W)
    mse_blend = _weighted_mse(B + w_opt * d, T, W)
    rel_improvement = float((mse_model - mse_blend) / max(mse_model, 1e-12))
    if mse_model <= mse_base * 1.05:
        return 1.0
    if rel_improvement < float(max(0.0, min_improvement)):
        return 1.0

    w_eff = float(np.clip(w_opt, 0.0, 1.0))
    if mse_model <= mse_base * float(max(1.0, emergency_ratio)):
        w_eff = max(float(np.clip(min_model_weight, 0.0, 1.0)), w_eff)
        mse_eff = _weighted_mse(B + w_eff * d, T, W)
        eff_improvement = float((mse_model - mse_eff) / max(mse_model, 1e-12))
        if eff_improvement < max(0.5 * float(min_improvement), 0.01):
            return 1.0
    return float(np.clip(w_eff, 0.0, 1.0))


def build_global_hybrid_components(
    pairs: Sequence[Tuple[str, np.ndarray, np.ndarray]],
    cfg: TrainConfig,
    base_model_fn,
    wavelet: str = "db4",
    level: int = 1,
    boundary: str = "wrap",
    decompose_fn: Callable[[np.ndarray], Sequence[np.ndarray]] | None = None,
) -> List[HybridComponent]:

    if not pairs:
        return []

    comp_values: List[List[np.ndarray]] = []
    if decompose_fn is None:
        def decompose_fn(y_arr: np.ndarray) -> Sequence[np.ndarray]:
            A_loc, D_loc = modwt_decompose_with_boundary(
                y_arr,
                wavelet=wavelet,
                level=level,
                boundary=boundary,
                check=True,
            )
            return [A_loc] + D_loc if len(D_loc) else [A_loc]

    for _, y_tr, _ in pairs:
        y = np.asarray(y_tr, float)
        if y.size < 2:
            continue
        comps = [np.asarray(comp, float) for comp in decompose_fn(y)]
        if not comps:
            comps = [y]
        if not comp_values:
            comp_values = [[] for _ in range(len(comps))]
        for idx, comp in enumerate(comps):
            if idx >= len(comp_values):
                comp_values.append([])
            comp_values[idx].append(np.asarray(comp, float))

    components: List[HybridComponent] = []
    H = cfg.horizon
    base_L = int(cfg.lookback)
    n_components = len(comp_values)
    for comp_idx, comp_series in enumerate(comp_values):
        if not comp_series:
            components.append(HybridComponent(None, 0.0, 1.0, None, per_series_scaling=True))
            continue
        Lmax_series = min((len(arr) - H for arr in comp_series), default=0)
        if Lmax_series <= 0:
            components.append(HybridComponent(None, 0.0, 1.0, None, per_series_scaling=True))
            continue
        L = _component_lookback(
            base_L,
            horizon=H,
            Lmax=int(Lmax_series),
            component_index=comp_idx,
            n_components=n_components,
        )
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
        cfg_local = replace(cfg, lookback=int(L))
        model = base_model_fn(cfg_local)
        trained = train_model(model, ds, cfg_local)
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
        component_names: Sequence[str] | None = None,
        enable_output_blend: bool = True,
        output_blend_min_alpha: float = 0.65,
        output_blend_hard_min_alpha: float = 0.45,
        output_blend_min_improvement: float = 0.08,
        output_blend_emergency_ratio: float = 1.4,
        component_blend_min_weight: float = 0.55,
        component_blend_min_improvement: float = 0.03,
        component_blend_emergency_ratio: float = 1.4,
    ):
        self.base_model_fn = base_model_fn
        self.cfg = cfg
        self.wavelet = wavelet
        self.level = level
        self.boundary = str(boundary).lower()
        self.components: List[HybridComponent] = []
        self.pretrained_components = pretrained_components
        self.seasonal_period = int(seasonal_period) if seasonal_period is not None else None
        self.component_names = tuple(component_names) if component_names is not None else None
        self.enable_output_blend = bool(enable_output_blend)
        self.output_blend_min_alpha = float(output_blend_min_alpha)
        self.output_blend_hard_min_alpha = float(output_blend_hard_min_alpha)
        self.output_blend_min_improvement = float(output_blend_min_improvement)
        self.output_blend_emergency_ratio = float(output_blend_emergency_ratio)
        self.component_blend_min_weight = float(component_blend_min_weight)
        self.component_blend_min_improvement = float(component_blend_min_improvement)
        self.component_blend_emergency_ratio = float(component_blend_emergency_ratio)
        self.component_names_: tuple[str, ...] = ()
        self.output_blend_kind: str = "none"
        self.output_blend_alpha: float = 1.0
        self.output_blend_period: int | None = None

    def _resolve_component_names(self, n_components: int) -> tuple[str, ...]:
        if self.component_names is not None:
            if len(self.component_names) != n_components:
                raise ValueError(
                    f"component_names length mismatch: got {len(self.component_names)}, expected {n_components}"
                )
            return tuple(self.component_names)
        if n_components <= 0:
            return ()
        return tuple(["A_J"] + [f"D_{j}" for j in range(1, n_components)])

    def _prepare_component(self, comp: np.ndarray, *, component_index: int, n_components: int) -> HybridComponent:
        Lmax = len(comp) - self.cfg.horizon
        min_windows = 8
        if Lmax <= 0:
            return HybridComponent(None, float(comp.mean()), float(comp.std() + 1e-8), None, per_series_scaling=False)
        L = _component_lookback(
            int(self.cfg.lookback),
            horizon=int(self.cfg.horizon),
            Lmax=int(Lmax),
            component_index=component_index,
            n_components=n_components,
        )
        ds = WindowDatasetStd(comp, L, self.cfg.horizon, stride=1, scale=True)
        if len(ds) < min_windows:
            mu, sd = ds.scaler
            return HybridComponent(None, float(mu), float(sd), None, per_series_scaling=False)
        mu, sd = ds.scaler
        cfg_local = replace(self.cfg, lookback=int(L))
        model = self.base_model_fn(cfg_local)
        trained = train_model(model, ds, cfg_local)
        component = HybridComponent(trained, mu, sd, L, per_series_scaling=False)
        component.blend_weight = _calibrate_blend_weight(
            component,
            np.asarray(comp, float),
            horizon=self.cfg.horizon,
            device=self.cfg.device,
            min_model_weight=self.component_blend_min_weight,
            min_improvement=self.component_blend_min_improvement,
            emergency_ratio=self.component_blend_emergency_ratio,
        )
        return component

    def _calibrate_output_blend(self, y: np.ndarray) -> None:
        y = np.asarray(y, float)
        H = int(self.cfg.horizon)
        self.output_blend_kind = "none"
        self.output_blend_alpha = 1.0
        self.output_blend_period = None
        if H <= 0 or y.size < (3 * H + 8):
            return

        y_hist = y[:-H]
        y_tgt = y[-H:]
        if y_hist.size < 8:
            return

        try:
            hyb = np.asarray(self.forecast(y_hist), float)
        except Exception:
            return

        seasonal_periods = ((self.seasonal_period,) if self.seasonal_period and self.seasonal_period > 1 else ())
        kind_best, alpha_best, period_best = _select_output_blend(
            hyb,
            y_hist=y_hist,
            y_tgt=y_tgt,
            seasonal_periods=seasonal_periods,
            min_hybrid_weight=self.output_blend_min_alpha,
            hard_min_hybrid_weight=self.output_blend_hard_min_alpha,
            min_improvement=self.output_blend_min_improvement,
            emergency_ratio=self.output_blend_emergency_ratio,
        )
        self.output_blend_kind = str(kind_best)
        self.output_blend_alpha = float(alpha_best)
        self.output_blend_period = int(period_best) if period_best else None

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
        self.component_names_ = self._resolve_component_names(len(comps))
        if self.pretrained_components is not None and len(self.pretrained_components) == len(comps):
            # Reuse globally trained components; no per-series training.
            self.components = self.pretrained_components
        else:
            self.components = [
                self._prepare_component(comp, component_index=idx, n_components=len(comps))
                for idx, comp in enumerate(comps)
            ]
        if self.enable_output_blend:
            self._calibrate_output_blend(y)
        else:
            self.output_blend_kind = "none"
            self.output_blend_alpha = 1.0
            self.output_blend_period = None
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
            pred = _neural_component_pred(component, np.asarray(comp, float), H, self.cfg.device)
            pred = _blend_with_persistence(pred, np.asarray(comp, float), H, component.blend_weight)
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
        yhat = _stabilize_reconstruction_boundary(y, yhat)

        if self.enable_output_blend and self.output_blend_kind != "none" and self.output_blend_alpha < 0.999:
            try:
                base = _baseline_forecast_raw(
                    np.asarray(y, float),
                    H,
                    kind=self.output_blend_kind,
                    period=self.output_blend_period,
                )
                yhat = (
                    float(self.output_blend_alpha) * np.asarray(yhat, float)
                    + (1.0 - float(self.output_blend_alpha)) * np.asarray(base, float)
                )
                yhat = _stabilize_reconstruction_boundary(y, np.asarray(yhat, float))
            except Exception:
                pass
        return np.asarray(yhat, float)

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
            names = self.component_names_ or self._resolve_component_names(len(comps))
            return {names[0]: np.repeat(float(A[-1]) if A.size else 0.0, H).astype(float)}

        comp_preds: List[np.ndarray] = []
        for component, comp in zip(self.components, comps):
            pred = _neural_component_pred(component, np.asarray(comp, float), H, self.cfg.device)
            pred = _blend_with_persistence(pred, np.asarray(comp, float), H, component.blend_weight)
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

        names = self.component_names_ or self._resolve_component_names(len(comp_preds))
        out: dict[str, np.ndarray] = {names[0]: np.asarray(comp_preds[0], float)}
        for name, pred in zip(names[1:], comp_preds[1:]):
            out[name] = np.asarray(pred, float)
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
        seasonal_periods: Sequence[int] | None = None,
        boundary: str = "reflect",
        detail_anchor: bool = True,
        detail_transition_steps: int = 3,
        detail_dampen: bool = True,
        detail_dampen_tau: float = 3.0,
        residual_anchor: bool = True,
        residual_transition_steps: int = 6,
        enable_output_blend: bool = True,
        component_names: Sequence[str] | None = None,
        output_blend_min_alpha: float = 0.65,
        output_blend_hard_min_alpha: float = 0.45,
        output_blend_min_improvement: float = 0.08,
        output_blend_emergency_ratio: float = 1.4,
        component_blend_min_weight: float = 0.55,
        component_blend_min_improvement: float = 0.03,
        component_blend_emergency_ratio: float = 1.4,
    ) -> None:
        self.aj_model_fn = aj_model_fn
        self.detail_method = str(detail_method).lower()
        self.cfg = cfg
        self.wavelet = wavelet
        self.level = int(level)
        self.seasonal_period = int(seasonal_period) if seasonal_period is not None else None
        if seasonal_periods is None:
            self.seasonal_periods = (
                (self.seasonal_period,) if self.seasonal_period and self.seasonal_period > 1 else ()
            )
        else:
            self.seasonal_periods = tuple(
                sorted({int(p) for p in seasonal_periods if p is not None and int(p) > 1})
            )
            if self.seasonal_period is None and self.seasonal_periods:
                self.seasonal_period = self.seasonal_periods[0]
        self.boundary = str(boundary).lower()
        self.detail_anchor = bool(detail_anchor)
        self.detail_transition_steps = int(detail_transition_steps)
        self.detail_dampen = bool(detail_dampen)
        self.detail_dampen_tau = float(detail_dampen_tau)
        self.residual_anchor = bool(residual_anchor)
        self.residual_transition_steps = int(residual_transition_steps)
        self.enable_output_blend = bool(enable_output_blend)
        self.component_names = tuple(component_names) if component_names is not None else None
        self.output_blend_min_alpha = float(output_blend_min_alpha)
        self.output_blend_hard_min_alpha = float(output_blend_hard_min_alpha)
        self.output_blend_min_improvement = float(output_blend_min_improvement)
        self.output_blend_emergency_ratio = float(output_blend_emergency_ratio)
        self.component_blend_min_weight = float(component_blend_min_weight)
        self.component_blend_min_improvement = float(component_blend_min_improvement)
        self.component_blend_emergency_ratio = float(component_blend_emergency_ratio)
        self.component_names_: tuple[str, ...] = ()
        self.aj_component: HybridComponent | None = None
        self.detail_policies: list[VWDetailPolicy] = []
        self.detail_residual_scale: float = 1.0
        self.output_blend_kind: str = "none"
        self.output_blend_alpha: float = 1.0  # weight of hybrid forecast
        self.output_blend_period: int | None = None

    def _resolve_component_names(self, n_components: int) -> tuple[str, ...]:
        if self.component_names is not None:
            if len(self.component_names) != n_components:
                raise ValueError(
                    f"component_names length mismatch: got {len(self.component_names)}, expected {n_components}"
                )
            return tuple(self.component_names)
        if n_components <= 0:
            return ()
        return tuple(["A_J"] + [f"D_{j}" for j in range(1, n_components)])

    def _prepare_component(self, comp: np.ndarray) -> HybridComponent:
        Lmax = len(comp) - self.cfg.horizon
        min_windows = 8
        if Lmax <= 0:
            return HybridComponent(None, float(comp.mean()), float(comp.std() + 1e-8), None)
        L = _component_lookback(
            int(self.cfg.lookback),
            horizon=int(self.cfg.horizon),
            Lmax=int(Lmax),
            component_index=0,
            n_components=1,
        )
        ds = WindowDatasetStd(comp, L, self.cfg.horizon, stride=1, scale=True)
        if len(ds) < min_windows:
            mu, sd = ds.scaler
            return HybridComponent(None, float(mu), float(sd), None)
        mu, sd = ds.scaler
        cfg_local = replace(self.cfg, lookback=int(L))
        model = self.aj_model_fn(cfg_local)
        trained = train_model(model, ds, cfg_local)
        component = HybridComponent(trained, mu, sd, L)
        component.blend_weight = _calibrate_blend_weight(
            component,
            np.asarray(comp, float),
            horizon=self.cfg.horizon,
            device=self.cfg.device,
            min_model_weight=self.component_blend_min_weight,
            min_improvement=self.component_blend_min_improvement,
            emergency_ratio=self.component_blend_emergency_ratio,
        )
        return component

    def _detail_candidate_forecast(
        self,
        comp_hist: np.ndarray,
        H: int,
        *,
        method: str,
        seasonal_period: int | None,
    ) -> np.ndarray:
        comp_hist = np.asarray(comp_hist, float)
        if method == "zero":
            return np.zeros(H, dtype=float)
        if method == "last":
            return _persistence_forecast(comp_hist, H)
        if method == "ets":
            seasonal_periods = None
            seasonal = None
            if seasonal_period and int(seasonal_period) > 1:
                seasonal_periods = int(seasonal_period)
                seasonal = "add"
            pred = ets_forecast(comp_hist, H, seasonal_periods=seasonal_periods, trend=None, seasonal=seasonal)
            return np.asarray(pred, float)
        if method == "arima_auto":
            return np.asarray(auto_arima_forecast(comp_hist, H), float)
        raise ValueError(f"Unknown detail candidate method '{method}'")

    def _choose_detail_policy(self, comp: np.ndarray) -> VWDetailPolicy:
        comp = np.asarray(comp, float)
        H = int(self.cfg.horizon)
        if H <= 0 or comp.size < H + 8:
            amp = float(3.0 * np.std(comp[-min(comp.size, 4 * max(H, 1)) :])) if comp.size else None
            return VWDetailPolicy(method="zero", seasonal_period=None, blend_zero=1.0, amp_limit=amp)

        hist = comp[:-H]
        tgt = comp[-H:]
        if hist.size < max(8, H):
            amp = float(3.0 * np.std(comp[-min(comp.size, 4 * max(H, 1)) :]))
            return VWDetailPolicy(method="last", seasonal_period=None, blend_zero=1.0, amp_limit=amp)

        tau = max(1.5, float(min(H, 8)) / 2.0)
        w = np.exp(-np.arange(H, dtype=float) / tau)
        w = w / (np.mean(w) + 1e-12)

        recent = comp[-min(comp.size, max(16, 4 * H)) :]
        recent_std = float(np.std(recent) + 1e-8)
        recent_abs_q = float(np.quantile(np.abs(recent), 0.95)) if recent.size else 0.0
        amp_limit = float(max(3.0 * recent_std, 1.5 * recent_abs_q, 1e-6))

        seasonal_candidates: list[int] = []
        for p in self.seasonal_periods:
            acf_p = abs(_acf_lag_score(hist, int(p)))
            if hist.size >= 2 * int(p) and acf_p >= 0.15 and recent_std > 1e-5:
                seasonal_candidates.append(int(p))

        preferred = "arima_auto" if self.detail_method in {"arima_auto", "auto_arima"} else "ets"
        candidate_specs: list[tuple[str, int | None]] = [("zero", None), ("last", None)]
        if preferred == "ets":
            candidate_specs.append(("ets", None))
            for p in seasonal_candidates:
                candidate_specs.append(("ets", p))
        else:
            candidate_specs.append(("arima_auto", None))
            candidate_specs.append(("ets", None))
            for p in seasonal_candidates:
                candidate_specs.append(("ets", p))

        best_policy = VWDetailPolicy(method="zero", seasonal_period=None, blend_zero=1.0, amp_limit=amp_limit)
        best_score = np.inf

        for method, seasonal_period in candidate_specs:
            try:
                pred_raw = self._detail_candidate_forecast(
                    hist, H, method=method, seasonal_period=seasonal_period
                )
            except Exception:
                continue
            pred_raw = _clip_detail_amplitude(pred_raw, hist, amp_limit)
            zeros = np.zeros(H, dtype=float)
            # Blend against zero to suppress unstable detail forecasts.
            d = pred_raw - zeros
            n = tgt - zeros
            denom = float(np.sum(w * d * d))
            if denom <= 1e-12:
                alpha_opt = 1.0 if method in {"zero", "last"} else 0.0
            else:
                alpha_opt = float(np.clip(np.sum(w * d * n) / denom, 0.0, 1.0))

            candidates_alpha = [1.0, alpha_opt]
            # extra conservative grid for noisy details
            candidates_alpha.extend([0.0, 0.25, 0.5, 0.75])
            for alpha in candidates_alpha:
                policy = VWDetailPolicy(
                    method=method,
                    seasonal_period=(int(seasonal_period) if seasonal_period else None),
                    blend_zero=float(np.clip(alpha, 0.0, 1.0)),
                    amp_limit=amp_limit,
                )
                try:
                    # Score the *actual* forecast path (with damping/anchoring/clipping),
                    # otherwise policy selection and inference use different objectives.
                    pred = self._forecast_detail(hist, policy=policy)
                except Exception:
                    continue
                score = _weighted_rmse(pred, tgt, w)
                if score < best_score:
                    best_score = score
                    best_policy = policy
        return best_policy

    def _fit_detail_policies(self, details: Sequence[np.ndarray]) -> None:
        self.detail_policies = [self._choose_detail_policy(np.asarray(d, float)) for d in details]

    def _calibrate_detail_residual_scale(self, details: Sequence[np.ndarray]) -> None:
        H = int(self.cfg.horizon)
        if H <= 0 or not details:
            self.detail_residual_scale = 1.0
            return
        preds: list[np.ndarray] = []
        tgts: list[np.ndarray] = []
        for idx, d in enumerate(details):
            d = np.asarray(d, float)
            if d.size < H + 8:
                continue
            hist = d[:-H]
            tgt = d[-H:]
            policy = self.detail_policies[idx] if idx < len(self.detail_policies) else VWDetailPolicy()
            try:
                pred = self._forecast_detail(hist, policy=policy)
            except Exception:
                continue
            preds.append(np.asarray(pred, float))
            tgts.append(np.asarray(tgt, float))
        if not preds:
            self.detail_residual_scale = 1.0
            return
        P = np.sum(np.stack(preds, 0), axis=0)
        T = np.sum(np.stack(tgts, 0), axis=0)
        tau = max(1.5, float(min(H, 8)) / 2.0)
        w = np.exp(-np.arange(H, dtype=float) / tau)
        w = w / (np.mean(w) + 1e-12)
        denom = float(np.sum(w * P * P))
        if denom <= 1e-12:
            self.detail_residual_scale = 1.0
            return
        scale = float(np.clip(np.sum(w * P * T) / denom, 0.0, 1.0))
        self.detail_residual_scale = scale

    def _baseline_forecast_raw(self, y_hist: np.ndarray, H: int, *, kind: str, period: int | None) -> np.ndarray:
        return _baseline_forecast_raw(y_hist, H, kind=kind, period=period)

    def _calibrate_output_blend(self, y: np.ndarray) -> None:
        y = np.asarray(y, float)
        H = int(self.cfg.horizon)
        self.output_blend_kind = "none"
        self.output_blend_alpha = 1.0
        self.output_blend_period = None
        if H <= 0 or y.size < (3 * H + 8):
            return

        y_hist = y[:-H]
        y_tgt = y[-H:]
        if y_hist.size < 8:
            return

        try:
            hyb = np.asarray(self.forecast(y_hist), float)
        except Exception:
            return

        kind_best, alpha_best, period_best = _select_output_blend(
            hyb,
            y_hist=y_hist,
            y_tgt=y_tgt,
            seasonal_periods=self.seasonal_periods,
            min_hybrid_weight=self.output_blend_min_alpha,
            hard_min_hybrid_weight=self.output_blend_hard_min_alpha,
            min_improvement=self.output_blend_min_improvement,
            emergency_ratio=self.output_blend_emergency_ratio,
        )
        self.output_blend_kind = str(kind_best)
        self.output_blend_alpha = float(alpha_best)
        self.output_blend_period = int(period_best) if period_best else None

    def fit(self, y, *, components_override: Optional[List[np.ndarray]] = None):
        y = np.asarray(y, float)
        if components_override is None:
            A, D = modwt_decompose_with_boundary(
                y, wavelet=self.wavelet, level=self.level, boundary=self.boundary, check=True
            )
        else:
            comps = [np.asarray(c, float) for c in components_override]
            A = comps[0] if comps else y
            D = comps[1:] if len(comps) > 1 else []
        self.component_names_ = self._resolve_component_names(1 + len(D))
        self._fit_detail_policies(D)
        self._calibrate_detail_residual_scale(D)
        self.aj_component = self._prepare_component(A)
        if self.enable_output_blend:
            self._calibrate_output_blend(y)
        else:
            self.output_blend_kind = "none"
            self.output_blend_alpha = 1.0
            self.output_blend_period = None
        return self

    def _forecast_neural(self, component: HybridComponent, comp: np.ndarray) -> np.ndarray:
        H = self.cfg.horizon
        pred = _neural_component_pred(component, np.asarray(comp, float), H, self.cfg.device)
        pred = _blend_with_persistence(pred, np.asarray(comp, float), H, component.blend_weight)
        return pred.astype(float, copy=False)

    def _forecast_detail(self, comp: np.ndarray, *, policy: VWDetailPolicy | None = None) -> np.ndarray:
        H = self.cfg.horizon
        comp = np.asarray(comp, float)
        if comp.size == 0:
            return np.zeros(H, dtype=float)
        pol = policy or VWDetailPolicy(method="zero", seasonal_period=None, blend_zero=1.0, amp_limit=None)
        pred = self._detail_candidate_forecast(
            comp,
            H,
            method=pol.method,
            seasonal_period=pol.seasonal_period,
        )
        pred = np.asarray(pred, float)
        if pred.size > H:
            pred = pred[:H]
        if pred.size < H:
            pred = np.pad(pred, (0, H - pred.size), mode="edge")

        out = pred.astype(float, copy=True)
        if pol.blend_zero < 0.999:
            out *= float(np.clip(pol.blend_zero, 0.0, 1.0))
        out = _clip_detail_amplitude(out, comp, pol.amp_limit)

        if self.detail_dampen and H > 0 and (pol.seasonal_period is None) and pol.method not in {"zero", "last"}:
            tau = max(0.5, float(self.detail_dampen_tau))
            w = np.exp(-np.arange(H, dtype=float) / tau)
            out = out * w

        if (not self.detail_anchor) or (pol.seasonal_period is not None) or comp.size < 2 or H <= 0:
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
        used_policies: List[VWDetailPolicy] = []
        for idx, dj in enumerate(D):
            policy = self.detail_policies[idx] if idx < len(self.detail_policies) else None
            pred = self._forecast_detail(np.asarray(dj, float), policy=policy)
            if policy is not None:
                used_policies.append(policy)
            if pred.size > H:
                pred = pred[:H]
            if pred.size < H:
                pred = np.pad(pred, (0, H - pred.size), mode="edge")
            detail_preds.append(pred.astype(float, copy=False))

        if detail_preds:
            resid_pred = np.sum(np.stack(detail_preds, 0), axis=0) * float(self.detail_residual_scale)
        else:
            resid_pred = np.zeros(H, dtype=float)

        has_seasonal_detail = any((p.seasonal_period is not None and p.blend_zero > 0.15) for p in used_policies)
        if self.residual_anchor and (not has_seasonal_detail) and y.size > 0 and A.size > 0 and H > 0:
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
        yhat = _stabilize_reconstruction_boundary(y, yhat)

        if self.enable_output_blend and self.output_blend_kind != "none" and self.output_blend_alpha < 0.999:
            try:
                base = self._baseline_forecast_raw(
                    np.asarray(y, float),
                    H,
                    kind=self.output_blend_kind,
                    period=self.output_blend_period,
                )
                yhat = (
                    float(self.output_blend_alpha) * np.asarray(yhat, float)
                    + (1.0 - float(self.output_blend_alpha)) * np.asarray(base, float)
                )
                yhat = _stabilize_reconstruction_boundary(y, np.asarray(yhat, float))
            except Exception:
                pass
        return np.asarray(yhat, float)

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
        used_policies: List[VWDetailPolicy] = []
        for idx, dj in enumerate(D):
            policy = self.detail_policies[idx] if idx < len(self.detail_policies) else None
            pred = self._forecast_detail(np.asarray(dj, float), policy=policy)
            if policy is not None:
                used_policies.append(policy)
            if pred.size > H:
                pred = pred[:H]
            if pred.size < H:
                pred = np.pad(pred, (0, H - pred.size), mode="edge")
            detail_preds.append(pred.astype(float, copy=False))

        if detail_preds:
            resid_pred = (np.sum(np.stack(detail_preds, 0), axis=0) * float(self.detail_residual_scale)).astype(float, copy=True)
        else:
            resid_pred = np.zeros(H, dtype=float)

        has_seasonal_detail = any((p.seasonal_period is not None and p.blend_zero > 0.15) for p in used_policies)
        if self.residual_anchor and (not has_seasonal_detail) and y.size > 0 and A.size > 0 and H > 0:
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
            raw = np.sum(np.stack(detail_preds, 0), axis=0) * float(self.detail_residual_scale)
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

        names = self.component_names_ or self._resolve_component_names(1 + len(details))
        out: dict[str, np.ndarray] = {names[0]: np.asarray(aj_pred, float)}
        for name, pred in zip(names[1:], details):
            out[name] = np.asarray(pred, float)
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
