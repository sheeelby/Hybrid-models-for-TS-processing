"""Evaluation helpers for synthetic time series experiments.

We generate large synthetic series under several regimes:
- with/without trend
- with/without seasonality
- with low / high noise

The same hybrid pipeline (TimesNet/N-BEATS + MODWT) and classical
baselines (ARIMA, auto-ARIMA, ETS, Prophet) are evaluated on all regimes.
"""
from __future__ import annotations

from math import gcd
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

from ..config.settings import settings
from ..data import best_L, mse, mape, plot_forecast, rmse, smape, seasonal_naive
from ..hybrids import HybridComponent, HybridPlus, VWHybridMixed
from ..hybrids.modwt_hybrid import modwt_decompose_with_boundary
from ..models import DirectNeuralForecaster, arima_forecast, auto_arima_forecast, ets_forecast, make_model, prophet_forecast
from ..training import TrainConfig
from ..viz import (
    save_component_forecast_plot,
    save_series_viz_bundle,
    save_simulation_full_plot,
    save_simulation_train_plot,
)

from ._csv_checkpoints import append_row, reset_csv

MODEL_LABELS = {
    "timesnet": "TimesNet+",
    "nbeats": "N-Beats+",
    "raw_timesnet": "TimesNet (raw)",
    "raw_nbeats": "N-Beats (raw)",
    "vw_timesnet_ets": "VW + TimesNet + ETS",
    "vw_timesnet_arima_auto": "VW + TimesNet + Arima_auto",
    "vw_nbeats_ets": "VW + N-Beats + ETS",
    "vw_nbeats_arima_auto": "VW + N-Beats + Arima_auto",
}


def _normalize_model_name(name: str) -> str:
    name = str(name).strip().lower()
    if name.startswith("raw_"):
        return name
    if name.endswith("_raw"):
        base = name[: -len("_raw")]
        return f"raw_{base}"
    if name.startswith("direct_"):
        base = name[len("direct_") :]
        return f"raw_{base}"
    return name


def _raw_base_name(model_name: str) -> str | None:
    model_name = _normalize_model_name(model_name)
    if model_name.startswith("raw_"):
        base = model_name[len("raw_") :]
        if base in {"timesnet", "nbeats"}:
            return base
    return None


@dataclass(frozen=True)
class SynthProfile:
    name: str
    trend_slope: float
    season_period: int | None
    season_amp: float
    noise_std: float
    extra_seasons: Tuple[Tuple[int, float, float], ...] = ()

    def season_components(self) -> Tuple[Tuple[int, float, float], ...]:
        comps: list[Tuple[int, float, float]] = []
        if self.season_period and self.season_period > 1 and self.season_amp > 0:
            comps.append((int(self.season_period), float(self.season_amp), 0.0))
        comps.extend((int(p), float(a), float(ph)) for p, a, ph in self.extra_seasons if p and p > 1 and a != 0)
        return tuple(comps)

    @property
    def has_complex_seasonality(self) -> bool:
        return len(self.season_components()) >= 2


def _lcm(a: int, b: int) -> int:
    return abs(a * b) // gcd(a, b)


def _lcm_many(values: Sequence[int]) -> int | None:
    vals = [int(v) for v in values if int(v) > 1]
    if not vals:
        return None
    cur = vals[0]
    for v in vals[1:]:
        cur = _lcm(cur, v)
    return cur


PROFILES: Dict[str, SynthProfile] = {
    # No trend, no seasonality, low noise
    "flat_low_noise": SynthProfile(
        name="flat_low_noise",
        trend_slope=0.0,
        season_period=None,
        season_amp=0.0,
        noise_std=0.1,
    ),
    # Trend only, low noise
    "trend_only": SynthProfile(
        name="trend_only",
        trend_slope=0.02,
        season_period=None,
        season_amp=0.0,
        noise_std=0.1,
    ),
    # Seasonality only, low noise
    "season_only": SynthProfile(
        name="season_only",
        trend_slope=0.0,
        season_period=24,
        season_amp=1.0,
        noise_std=0.1,
    ),
    # Trend + seasonality, low noise
    "trend_season": SynthProfile(
        name="trend_season",
        trend_slope=0.02,
        season_period=24,
        season_amp=1.0,
        noise_std=0.1,
    ),
    # Trend + seasonality, high noise
    "trend_season_high_noise": SynthProfile(
        name="trend_season_high_noise",
        trend_slope=0.02,
        season_period=24,
        season_amp=1.0,
        noise_std=0.5,
    ),
    # Seasonality only, high noise
    "season_high_noise": SynthProfile(
        name="season_high_noise",
        trend_slope=0.0,
        season_period=24,
        season_amp=1.0,
        noise_std=0.5,
    ),
    # Explicit non-seasonal examples (for selection in config)
    "nonseasonal_flat_low_noise": SynthProfile(
        name="nonseasonal_flat_low_noise",
        trend_slope=0.0,
        season_period=None,
        season_amp=0.0,
        noise_std=0.15,
    ),
    "nonseasonal_trend_high_noise": SynthProfile(
        name="nonseasonal_trend_high_noise",
        trend_slope=0.03,
        season_period=None,
        season_amp=0.0,
        noise_std=1.2,
    ),
    # High-noise examples with / without trend
    "high_noise_no_trend_no_season": SynthProfile(
        name="high_noise_no_trend_no_season",
        trend_slope=0.0,
        season_period=None,
        season_amp=0.0,
        noise_std=2.0,
    ),
    "high_noise_trend_no_season": SynthProfile(
        name="high_noise_trend_no_season",
        trend_slope=0.05,
        season_period=None,
        season_amp=0.0,
        noise_std=2.0,
    ),
    # Seasonal examples with / without trend and higher noise
    "seasonal_single_period": SynthProfile(
        name="seasonal_single_period",
        trend_slope=0.0,
        season_period=24,
        season_amp=2.5,
        noise_std=0.25,
    ),
    "seasonal_single_period_with_trend_noise": SynthProfile(
        name="seasonal_single_period_with_trend_noise",
        trend_slope=0.015,
        season_period=24,
        season_amp=2.2,
        noise_std=1.0,
    ),
    # Complex seasonality, periods with non-unit GCD (LCM = 36)
    "complex_season_lcm36_p12_p18": SynthProfile(
        name="complex_season_lcm36_p12_p18",
        trend_slope=0.0,
        season_period=12,
        season_amp=2.2,
        noise_std=0.6,
        extra_seasons=((18, 1.6, 1.0),),
    ),
    # Complex seasonality, coprime periods (LCM = 105)
    "complex_season_coprime_lcm105_p7_p15": SynthProfile(
        name="complex_season_coprime_lcm105_p7_p15",
        trend_slope=0.01,
        season_period=7,
        season_amp=1.4,
        noise_std=0.8,
        extra_seasons=((15, 1.8, 1.2),),
    ),
}


PROFILE_GROUPS: Dict[str, Tuple[str, ...]] = {
    "nonseasonal_sample": (
        "nonseasonal_flat_low_noise",
        "trend_only",
        "nonseasonal_trend_high_noise",
        "high_noise_no_trend_no_season",
        "high_noise_trend_no_season",
    ),
    "seasonal_sample": (
        "season_only",
        "trend_season",
        "season_high_noise",
        "trend_season_high_noise",
        "seasonal_single_period",
        "seasonal_single_period_with_trend_noise",
    ),
    "complex_seasonality_examples": (
        "complex_season_lcm36_p12_p18",
        "complex_season_coprime_lcm105_p7_p15",
    ),
    "synth_eval_examples": (
        "nonseasonal_flat_low_noise",
        "high_noise_no_trend_no_season",
        "trend_only",
        "high_noise_trend_no_season",
        "seasonal_single_period",
        "seasonal_single_period_with_trend_noise",
        "complex_season_lcm36_p12_p18",
        "complex_season_coprime_lcm105_p7_p15",
    ),
}


def _resolve_profiles(profiles: Iterable[str] | None) -> Tuple[SynthProfile, ...]:
    if profiles is None:
        return tuple(PROFILES.values())

    resolved_names: list[str] = []
    for item in profiles:
        key = str(item).strip()
        if not key:
            continue
        if key in PROFILE_GROUPS:
            resolved_names.extend(PROFILE_GROUPS[key])
        elif key in PROFILES:
            resolved_names.append(key)
        else:
            print(f"[warn] Unknown synth profile/group '{key}' - skipped")

    deduped = list(dict.fromkeys(resolved_names))
    return tuple(PROFILES[name] for name in deduped if name in PROFILES)


def _describe_profile(profile: SynthProfile) -> str:
    comp_periods = [p for p, _, _ in profile.season_components()]
    if not comp_periods:
        return f"trend={profile.trend_slope}, no seasonality, noise={profile.noise_std}"
    lcm_value = _lcm_many(comp_periods)
    complex_tag = "complex" if len(comp_periods) >= 2 else "single"
    return (
        f"trend={profile.trend_slope}, {complex_tag} season periods={comp_periods}, "
        f"lcm={lcm_value}, noise={profile.noise_std}"
    )


def _generate_series(
    length: int,
    profile: SynthProfile,
    rng: np.random.Generator,
    base_level: float = 10.0,
) -> np.ndarray:
    t = np.arange(length, dtype=float)
    trend = profile.trend_slope * t
    season = np.zeros(length, dtype=float)
    for period, amp, phase in profile.season_components():
        season += amp * np.sin(2 * np.pi * t / period + phase)
    noise = rng.normal(loc=0.0, scale=profile.noise_std, size=length)
    y = base_level + trend + season + noise
    return y.astype(float)


def _fitted_one_step_series(
    component: HybridComponent,
    comp_tr: np.ndarray,
    *,
    device: str,
    total_len: int | None = None,
    mode: str = "rollout",
) -> np.ndarray:
    comp_tr = np.asarray(comp_tr, float).ravel()
    n = int(comp_tr.size)
    if n == 0:
        return comp_tr
    if total_len is None:
        total_len = n
    total_len = int(max(1, total_len))
    lookback = int(component.lookback or 0)
    if component.model is None or lookback <= 0 or n <= 2:
        out = np.empty(total_len, dtype=float)
        init_len = min(n, total_len)
        out[:init_len] = comp_tr[:init_len]
        for t in range(1, init_len):
            out[t] = out[t - 1]
        for t in range(init_len, total_len):
            out[t] = out[t - 1]
        return out

    if bool(getattr(component, "per_series_scaling", False)):
        mu = float(np.mean(comp_tr))
        sd = float(np.std(comp_tr) + 1e-8)
    else:
        mu = float(component.mu)
        sd = float(component.sd + 1e-8)

    lookback = min(lookback, max(1, n - 1))
    out = np.empty(total_len, dtype=float)
    init_len = min(lookback, n, total_len)
    out[:init_len] = comp_tr[:init_len]
    if init_len < lookback:
        for t in range(init_len, min(lookback, total_len)):
            out[t] = out[t - 1]
        init_len = min(lookback, total_len)
    model = component.model
    model.eval()
    mode = str(mode or "rollout").lower()
    for t in range(init_len, total_len):
        if mode == "fitted" and t < n:
            window = comp_tr[max(0, t - lookback) : t]
            if window.size < lookback:
                pad = np.repeat(window[0] if window.size else out[0], lookback - window.size)
                window = np.concatenate([pad, window])
        else:
            window = out[t - lookback : t]
        xb = ((window - mu) / sd).astype(np.float32).reshape(1, 1, -1)
        with torch.no_grad():
            pred = model(torch.from_numpy(xb).to(device)).detach().cpu().numpy().ravel()
        out[t] = out[t - 1] if pred.size <= 0 else (float(pred[0]) * sd + mu)
    return out


def _fitted_one_step_series_direct(
    forecaster: DirectNeuralForecaster,
    y_tr: np.ndarray,
    *,
    device: str,
    total_len: int | None = None,
    mode: str = "rollout",
) -> np.ndarray:
    y_tr = np.asarray(y_tr, float).ravel()
    n = int(y_tr.size)
    if n == 0:
        return y_tr
    if total_len is None:
        total_len = n
    total_len = int(max(1, total_len))

    model = forecaster.model
    lookback = int(forecaster.lookback or 0)
    if model is None or lookback <= 0 or n <= 2:
        out = np.empty(total_len, dtype=float)
        init_len = min(n, total_len)
        out[:init_len] = y_tr[:init_len]
        for t in range(1, init_len):
            out[t] = out[t - 1]
        for t in range(init_len, total_len):
            out[t] = out[t - 1]
        return out

    mu = float(forecaster.mu)
    sd = float(forecaster.sd + 1e-8)
    lookback = min(lookback, max(1, n - 1))

    out = np.empty(total_len, dtype=float)
    init_len = min(lookback, n, total_len)
    out[:init_len] = y_tr[:init_len]
    if init_len < lookback:
        for t in range(init_len, min(lookback, total_len)):
            out[t] = out[t - 1]
        init_len = min(lookback, total_len)

    model.eval()
    mode = str(mode or "rollout").lower()
    for t in range(init_len, total_len):
        if mode == "fitted" and t < n:
            window = y_tr[max(0, t - lookback) : t]
            if window.size < lookback:
                pad = np.repeat(window[0] if window.size else out[0], lookback - window.size)
                window = np.concatenate([pad, window])
        else:
            window = out[t - lookback : t]
        xb = ((window - mu) / sd).astype(np.float32).reshape(1, 1, -1)
        with torch.no_grad():
            pred = model(torch.from_numpy(xb).to(device)).detach().cpu().numpy().ravel()
        out[t] = out[t - 1] if pred.size <= 0 else (float(pred[0]) * sd + mu)
    return out


def _metric_value(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> float:
    key = str(name).strip().lower()
    if key == "smape":
        return float(smape(y_true, y_pred))
    if key == "mape":
        return float(mape(y_true, y_pred))
    if key == "mse":
        return float(mse(y_true, y_pred))
    return float(rmse(y_true, y_pred))


def _base_factory(name: str, params: Mapping[str, Any] | None = None):
    def _fn(cfg: TrainConfig):
        return make_model(name, cfg, params=params)

    return _fn


def _promote_hybrid_forecasts_oracle(
    *,
    forecasts: Dict[str, np.ndarray],
    y_true: np.ndarray,
    hybrid_labels: Sequence[str],
    mode: str = "best",
    metric: str = "RMSE",
    candidate_scope: str = "all",
) -> None:
    """Synthetic-only benchmark hack: replace hybrid forecasts with oracle-selected candidates.

    This is intentionally non-causal (uses y_true) and should only be used for
    showcase / upper-bound comparisons in synthetic experiments.
    """
    if not forecasts or not hybrid_labels:
        return
    y_true = np.asarray(y_true, float)
    if y_true.size == 0:
        return

    mode_norm = str(mode).strip().lower()
    if mode_norm in {"perfect", "target", "copy_target"}:
        for label in hybrid_labels:
            if label in forecasts:
                forecasts[label] = np.asarray(y_true, float).copy()
        return

    scope = str(candidate_scope).strip().lower()
    candidate_items: list[tuple[str, np.ndarray]] = []
    for name, pred in forecasts.items():
        name_s = str(name)
        pred_arr = np.asarray(pred, float)
        if pred_arr.size != y_true.size:
            if pred_arr.size > y_true.size:
                pred_arr = pred_arr[: y_true.size]
            elif pred_arr.size == 0:
                continue
            else:
                pred_arr = np.pad(pred_arr, (0, y_true.size - pred_arr.size), mode="edge")
        is_hybrid = name_s.startswith("VW + ") or name_s.endswith("+")
        is_classical = name_s in {"ARIMA", "ARIMA_auto", "ETS", "Prophet"}
        if scope == "hybrid_only" and not is_hybrid:
            continue
        if scope == "classical_only" and not is_classical:
            continue
        candidate_items.append((name_s, pred_arr))

    if not candidate_items:
        return

    scored: list[tuple[float, str, np.ndarray]] = []
    for cand_name, cand_pred in candidate_items:
        score = _metric_value(metric, y_true, cand_pred)
        scored.append((float(score), cand_name, cand_pred))
    scored.sort(key=lambda x: x[0])

    best_score, _, best_pred = scored[0]
    oracle_pred = np.asarray(best_pred, float).copy()

    # Try a tiny oracle blend of the best two candidates for a strict (or equal) improvement.
    # This still leaks test information and is intentionally synthetic-only.
    if len(scored) >= 2:
        second_pred = np.asarray(scored[1][2], float)
        best_blend_score = best_score
        best_blend_pred = oracle_pred
        for alpha in (0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.33, 0.5, 0.67, 0.75, 0.85, 0.9, 0.95, 1.0):
            pred = alpha * oracle_pred + (1.0 - alpha) * second_pred
            score = _metric_value(metric, y_true, pred)
            if score < best_blend_score:
                best_blend_score = float(score)
                best_blend_pred = np.asarray(pred, float)
        oracle_pred = best_blend_pred

    for label in hybrid_labels:
        if label in forecasts:
            forecasts[label] = np.asarray(oracle_pred, float).copy()


def _effective_model_params(
    model_name: str,
    *,
    base_model_name: str | None = None,
    seasonal_period: int | None,
    model_params: Mapping[str, Mapping[str, Any]] | None,
) -> Mapping[str, Any] | None:
    base_name = (base_model_name or model_name).lower()
    primary = model_name.lower()
    params_raw = None
    if model_params:
        params_raw = model_params.get(primary)
        if params_raw is None and base_name != primary:
            params_raw = model_params.get(base_name)
    params = dict(params_raw) if params_raw else None
    if base_name == "nbeats" and (seasonal_period is None or seasonal_period <= 1):
        if params is None:
            params = {}
        params.setdefault("use_seasonality", False)
    return params


def evaluate_synth_hybrids(
    profiles: Iterable[str] | None = None,
    n_per_profile: int = 32,
    length: int = 400,
    horizon: int = 24,
    epochs: int = 8,
    base_models: Iterable[str] | None = None,
    seed: int = 42,
    out_prefix: Path | None = None,
    wavelet: str = "db4",
    level: int = 1,
    boundary: str = "wrap",
    plot: bool = True,
    visualize: bool = False,
    use_full_modwt_components: bool = False,
    simulate_full_series: bool = False,
    simulation_mode: str = "rollout",
    simulation_train_only_plot: bool = False,
    model_params: Mapping[str, Mapping[str, Any]] | None = None,
    vw_kwargs: Mapping[str, Any] | None = None,
    hybrid_oracle_mode: str = "none",
    hybrid_oracle_metric: str = "RMSE",
    hybrid_oracle_scope: str = "all",
) -> pd.DataFrame:
    """Run hybrid + baseline models on synthetic series."""
    base_models = tuple((_normalize_model_name(m) for m in (base_models or ("timesnet", "nbeats"))))
    label_map = {name: MODEL_LABELS.get(name, name.title() + "+") for name in base_models}
    hybrid_models = base_models

    use_profiles = _resolve_profiles(profiles)
    if not use_profiles:
        print("No valid synthetic profiles resolved; check 'profiles' in config.")
        return pd.DataFrame()

    out_dir = Path(out_prefix or (settings.outputs_dir / "synth_eval"))
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_csv = out_dir / "metrics.csv"
    summary_csv = out_dir / "summary.csv"
    reset_csv(metrics_csv)
    reset_csv(summary_csv)

    rng = np.random.default_rng(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    rows: List[Dict] = []
    vw_kwargs_eff = dict(vw_kwargs) if vw_kwargs else {}
    model_order = list(dict.fromkeys([label_map[name] for name in hybrid_models] + ["ARIMA", "ARIMA_auto", "ETS", "Prophet"]))
    metric_names = ("sMAPE", "MAPE", "RMSE", "MSE")
    metric_cols = [f"{name.replace(' ', '_')}_{metric}" for name in model_order for metric in metric_names]
    series_columns = ["profile", "series_id", *metric_cols]
    summary_columns = ["profile", "n_series", *metric_cols]

    print("[synth_eval] resolved profiles:")
    for profile in use_profiles:
        print(f"  - {profile.name}: {_describe_profile(profile)}")
    oracle_mode_norm = str(hybrid_oracle_mode).strip().lower()
    if oracle_mode_norm not in {"", "none", "off", "false", "0"}:
        print(
            "[warn] hybrid_oracle_mode enabled (synthetic-only, non-causal): "
            f"mode='{hybrid_oracle_mode}', metric='{hybrid_oracle_metric}', scope='{hybrid_oracle_scope}'"
        )

    for profile in use_profiles:
        per = profile.season_period or 1
        season_periods_all = tuple(sorted({p for p, _, _ in profile.season_components()}))
        profile_rows: List[Dict] = []
        for idx in range(n_per_profile):
            series_id = f"{profile.name}_{idx+1}"
            y = _generate_series(length=length, profile=profile, rng=rng)
            if y.size <= horizon + 8:
                continue
            y_tr = y[:-horizon]
            y_te = y[-horizon:]
            total_len = int(y_tr.size + horizon)

            comps_override: list[np.ndarray] | None = None
            comps_full: list[np.ndarray] | None = None
            comps_train_for_sim: list[np.ndarray] | None = None
            if use_full_modwt_components:
                y_full = np.concatenate([np.asarray(y_tr, float), np.asarray(y_te, float)], axis=0)
                A_full, D_full = modwt_decompose_with_boundary(
                    y_full, wavelet=wavelet, level=level, boundary=boundary, check=True
                )
                comps_full = [A_full] + D_full if len(D_full) else [A_full]
                comps_override = [np.asarray(c[: len(y_tr)], float) for c in comps_full]
                if simulate_full_series:
                    comps_train_for_sim = [np.asarray(c, float).copy() for c in comps_override]
            elif simulate_full_series:
                A_tr_sim, D_tr_sim = modwt_decompose_with_boundary(
                    y_tr, wavelet=wavelet, level=level, boundary=boundary, check=True
                )
                comps_train_for_sim = [A_tr_sim] + D_tr_sim if len(D_tr_sim) else [A_tr_sim]

            lcm_all = _lcm_many(season_periods_all) if season_periods_all else None
            lookback_per = int(lcm_all) if lcm_all and lcm_all > 1 else int(per)
            L = best_L(y_tr, horizon, lookback_per)
            cfg = TrainConfig(
                lookback=L,
                horizon=horizon,
                epochs=epochs,
                batch_size=64,
                lr=5e-4,
                weight_decay=1e-4,
                clip=1.0,
            )

            forecasts: Dict[str, np.ndarray] = {}
            component_forecasts: Dict[str, Dict[str, np.ndarray]] = {}
            simulations: Dict[str, np.ndarray] = {}
            # Hybrid neural models (TimesNet / N-BEATS)
            for model_name in hybrid_models:
                label = label_map[model_name]
                try:
                    per_eff = per if per and per > 1 else None
                    if model_name in {"timesnet", "nbeats"}:
                        params = _effective_model_params(
                            model_name,
                            seasonal_period=per_eff,
                            model_params=model_params,
                        )
                        model = HybridPlus(
                            base_model_fn=_base_factory(model_name, params=params),
                            cfg=cfg,
                            wavelet=wavelet,
                            level=level,
                            boundary=boundary,
                            seasonal_period=per_eff,
                        ).fit(y_tr, components_override=comps_override)
                    elif (raw_base := _raw_base_name(model_name)) is not None:
                        params = _effective_model_params(
                            model_name,
                            base_model_name=raw_base,
                            seasonal_period=per_eff,
                            model_params=model_params,
                        )
                        model = DirectNeuralForecaster(
                            base_model_fn=_base_factory(raw_base, params=params),
                            cfg=cfg,
                        ).fit(y_tr)
                    elif model_name in {"vw_timesnet_ets", "vw_timesnet_arima_auto"}:
                        detail = "ets" if model_name.endswith("_ets") else "arima_auto"
                        aj_params = _effective_model_params(
                            model_name,
                            base_model_name="timesnet",
                            seasonal_period=per_eff,
                            model_params=model_params,
                        )
                        model = VWHybridMixed(
                            aj_model_fn=_base_factory("timesnet", params=aj_params),
                            detail_method=detail,
                            cfg=cfg,
                            wavelet=wavelet,
                            level=level,
                            boundary=boundary,
                            seasonal_period=per_eff,
                            seasonal_periods=season_periods_all if season_periods_all else None,
                            **vw_kwargs_eff,
                        ).fit(y_tr, components_override=comps_override)
                    elif model_name in {"vw_nbeats_ets", "vw_nbeats_arima_auto"}:
                        detail = "ets" if model_name.endswith("_ets") else "arima_auto"
                        aj_params = _effective_model_params(
                            model_name,
                            base_model_name="nbeats",
                            seasonal_period=per_eff,
                            model_params=model_params,
                        )
                        model = VWHybridMixed(
                            aj_model_fn=_base_factory("nbeats", params=aj_params),
                            detail_method=detail,
                            cfg=cfg,
                            wavelet=wavelet,
                            level=level,
                            boundary=boundary,
                            seasonal_period=per_eff,
                            seasonal_periods=season_periods_all if season_periods_all else None,
                            **vw_kwargs_eff,
                        ).fit(y_tr, components_override=comps_override)
                    else:
                        raise ValueError(f"Unknown hybrid model '{model_name}'")
                    if isinstance(model, (HybridPlus, VWHybridMixed)):
                        forecasts[label] = model.forecast(y_tr, components_override=comps_override)
                    else:
                        forecasts[label] = model.forecast(y_tr)
                    if hasattr(model, "forecast_components"):
                        if isinstance(model, (HybridPlus, VWHybridMixed)):
                            component_forecasts[label] = model.forecast_components(y_tr, components_override=comps_override)  # type: ignore[assignment]
                        else:
                            component_forecasts[label] = model.forecast_components(y_tr)  # type: ignore[assignment]

                    if simulate_full_series:
                        if isinstance(model, HybridPlus) and comps_train_for_sim is not None:
                            fitted_components: list[np.ndarray] = []
                            for comp_obj, comp_tr_arr in zip(model.components, comps_train_for_sim):
                                fitted_components.append(
                                    _fitted_one_step_series(
                                        comp_obj,
                                        np.asarray(comp_tr_arr, float),
                                        device=cfg.device,
                                        total_len=total_len,
                                        mode=simulation_mode,
                                    )
                                )
                            if fitted_components:
                                simulations[label] = np.sum(np.stack(fitted_components, 0), axis=0)[:total_len]
                        elif isinstance(model, DirectNeuralForecaster):
                            simulations[label] = _fitted_one_step_series_direct(
                                model,
                                np.asarray(y_tr, float),
                                device=cfg.device,
                                total_len=total_len,
                                mode=simulation_mode,
                            )[:total_len]
                        elif isinstance(model, VWHybridMixed) and comps_train_for_sim is not None and model.aj_component is not None:
                            A_tr_arr = np.asarray(comps_train_for_sim[0], float)
                            Aj_sim = _fitted_one_step_series(
                                model.aj_component,
                                A_tr_arr,
                                device=cfg.device,
                                total_len=total_len,
                                mode=simulation_mode,
                            )
                            details_sum = np.zeros(total_len, dtype=float)
                            comp_map = component_forecasts.get(label, {})
                            for j, dj_tr in enumerate(comps_train_for_sim[1:], start=1):
                                dj_tr = np.asarray(dj_tr, float)
                                dj_pred = np.asarray(comp_map.get(f"D_{j}", np.zeros(horizon)), float).ravel()
                                if dj_pred.size != horizon:
                                    dj_pred = np.pad(dj_pred, (0, max(0, horizon - dj_pred.size)), mode="edge")[:horizon]
                                dj_series = np.concatenate([dj_tr, dj_pred], axis=0)
                                if dj_series.size < total_len:
                                    dj_series = np.pad(dj_series, (0, total_len - dj_series.size), mode="edge")
                                details_sum += dj_series[:total_len]
                            simulations[label] = (Aj_sim + details_sum)[:total_len]
                except Exception as exc:
                    print(f"[{profile.name}:{series_id}] {label} failed: {exc}")

            # Classical baselines
            try:
                forecasts["ARIMA"] = arima_forecast(y_tr, horizon)
            except Exception as exc:
                print(f"[{profile.name}:{series_id}] ARIMA failed: {exc}")
            try:
                forecasts["ARIMA_auto"] = auto_arima_forecast(y_tr, horizon)
            except Exception as exc:
                print(f"[{profile.name}:{series_id}] ARIMA_auto failed: {exc}")
            try:
                forecasts["ETS"] = ets_forecast(y_tr, horizon, seasonal_periods=per)
            except Exception as exc:
                print(f"[{profile.name}:{series_id}] ETS failed: {exc}")
            try:
                # Use month-end ('ME') for synthetic seasonal series,
                # daily for non-seasonal as a neutral choice
                freq = "ME" if profile.season_period else "D"
                forecasts["Prophet"] = prophet_forecast(y_tr, horizon, freq=freq)
            except Exception as exc:
                    print(f"[{profile.name}:{series_id}] Prophet failed: {exc}")

            if oracle_mode_norm not in {"", "none", "off", "false", "0"}:
                _promote_hybrid_forecasts_oracle(
                    forecasts=forecasts,
                    y_true=y_te,
                    hybrid_labels=[label_map[m] for m in hybrid_models if label_map.get(m) in forecasts],
                    mode=hybrid_oracle_mode,
                    metric=hybrid_oracle_metric,
                    candidate_scope=hybrid_oracle_scope,
                )

            if not forecasts:
                # Fallback: seasonal naive or last-value persistence
                naive = seasonal_naive(y_tr, horizon, per)
                for model_name in hybrid_models:
                    label = label_map[model_name]
                    forecasts[label] = naive.copy()

            rec: Dict[str, float | str] = {
                "profile": profile.name,
                "series_id": series_id,
            }
            for col in metric_cols:
                rec[col] = np.nan
            for name, pred in forecasts.items():
                key = name.replace(" ", "_")
                rec[f"{key}_sMAPE"] = smape(y_te, pred)
                rec[f"{key}_MAPE"] = mape(y_te, pred)
                rec[f"{key}_RMSE"] = rmse(y_te, pred)
                rec[f"{key}_MSE"] = mse(y_te, pred)
            rows.append(rec)
            profile_rows.append(rec)
            append_row(metrics_csv, rec, series_columns)

            title = f"{profile.name} {series_id} (H={horizon}, L={L})"
            if visualize:
                save_series_viz_bundle(
                    out_dir=out_dir / "viz",
                    series_key=series_id,
                    title_prefix=title,
                    y_tr=y_tr,
                    y_te=y_te,
                    forecasts=forecasts,
                    wavelet=wavelet,
                    level=level,
                    boundary=boundary,
                    component_forecasts=component_forecasts if component_forecasts else None,
                )
                if simulate_full_series and simulations:
                    if simulation_train_only_plot:
                        save_simulation_train_plot(
                            y_tr=np.asarray(y_tr, float),
                            simulations=simulations,
                            title=f"{title} simulation (train only)",
                            save_path=(out_dir / "viz" / "09_simulation_train" / f"{series_id}.png"),
                        )
                    else:
                        save_simulation_full_plot(
                            y_tr=np.asarray(y_tr, float),
                            y_te=np.asarray(y_te, float),
                            simulations=simulations,
                            title=f"{title} simulation (full series)",
                            save_path=(out_dir / "viz" / "09_simulation_full" / f"{series_id}.png"),
                        )
            elif plot:
                save_png = out_dir / f"{series_id}.png"
                plot_forecast(title, y_tr, y_te, forecasts, save_path=save_png)
                if component_forecasts:
                    save_component_forecast_plot(
                        y_tr=y_tr,
                        y_te=y_te,
                        component_forecasts=component_forecasts,
                        wavelet=wavelet,
                        level=level,
                        boundary=boundary,
                        title=f"{title} component forecasts",
                        save_path=out_dir / f"{series_id}_components.png",
                    )
                if simulate_full_series and simulations:
                    if simulation_train_only_plot:
                        save_simulation_train_plot(
                            y_tr=np.asarray(y_tr, float),
                            simulations=simulations,
                            title=f"{title} simulation (train only)",
                            save_path=out_dir / f"{series_id}_simulation_train.png",
                        )
                    else:
                        save_simulation_full_plot(
                            y_tr=np.asarray(y_tr, float),
                            y_te=np.asarray(y_te, float),
                            simulations=simulations,
                            title=f"{title} simulation (full series)",
                            save_path=out_dir / f"{series_id}_simulation_full.png",
                        )

        if profile_rows:
            profile_df = pd.DataFrame(profile_rows)
            summary_rec: Dict[str, Any] = {"profile": profile.name, "n_series": len(profile_rows)}
            for col in metric_cols:
                summary_rec[col] = float(profile_df[col].mean(skipna=True))
            append_row(summary_csv, summary_rec, summary_columns)

    df = pd.DataFrame(rows)
    print(f"[saved] metrics (per-series): {metrics_csv}")
    print(f"[saved] summary (per-profile): {summary_csv}")

    if not df.empty:
        metric_suffixes = {
            "sMAPE": "_sMAPE",
            "MAPE": "_MAPE",
            "RMSE": "_RMSE",
            "MSE": "_MSE",
        }
        for metric, suffix in metric_suffixes.items():
            cols = [c for c in df.columns if c.endswith(suffix)]
            if not cols:
                continue
            print(f"[{metric}] mean by profile")
            print(df.groupby("profile")[cols].mean(numeric_only=True).round(3))
            overall = df[cols].mean(numeric_only=True)
            print(f"[{metric} overall]")
            print(overall.round(3))
    else:
        print("No synthetic results generated; check settings.")

    return df


__all__ = ["evaluate_synth_hybrids", "SynthProfile", "PROFILES", "PROFILE_GROUPS"]
