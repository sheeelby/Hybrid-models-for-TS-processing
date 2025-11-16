"""Evaluation helpers for synthetic time series experiments.

We generate large synthetic series under several regimes:
- with/without trend
- with/without seasonality
- with low / high noise

The same hybrid pipeline (TimesNet/N-BEATS/Helformer + MODWT) and classical
baselines (ARIMA, auto-ARIMA, ETS, Prophet) are evaluated on all regimes.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import torch

from ..config.settings import settings
from ..data import best_L, mse, mape, plot_forecast, rmse, smape, seasonal_naive
from ..hybrids import HybridPlus
from ..models import (
    HelformerAutoRegressor,
    arima_forecast,
    auto_arima_forecast,
    ets_forecast,
    make_model,
    prophet_forecast,
)
from ..training import TrainConfig, WindowDatasetStd, train_model


@dataclass(frozen=True)
class SynthProfile:
    name: str
    trend_slope: float
    season_period: int | None
    season_amp: float
    noise_std: float


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
}


def _generate_series(
    length: int,
    profile: SynthProfile,
    rng: np.random.Generator,
    base_level: float = 10.0,
) -> np.ndarray:
    t = np.arange(length, dtype=float)
    trend = profile.trend_slope * t
    if profile.season_period and profile.season_period > 1 and profile.season_amp > 0:
        season = profile.season_amp * np.sin(2 * np.pi * t / profile.season_period)
    else:
        season = 0.0
    noise = rng.normal(loc=0.0, scale=profile.noise_std, size=length)
    y = base_level + trend + season + noise
    return y.astype(float)


def _base_factory(name: str):
    def _fn(cfg: TrainConfig):
        return make_model(name, cfg)

    return _fn


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
    plot: bool = True,
) -> pd.DataFrame:
    """Run hybrid + baseline models on synthetic series."""
    base_models = tuple((m.lower() for m in (base_models or ("timesnet", "nbeats", "helformer"))))
    label_map = {name: name.title() + "+" for name in base_models if name != "helformer"}

    use_profiles: Tuple[SynthProfile, ...]
    if profiles is None:
        use_profiles = tuple(PROFILES.values())
    else:
        use_profiles = tuple(PROFILES[p] for p in profiles if p in PROFILES)

    out_dir = Path(out_prefix or (settings.outputs_dir / "synth_eval"))
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    rows: List[Dict] = []

    for profile in use_profiles:
        per = profile.season_period or 1
        for idx in range(n_per_profile):
            series_id = f"{profile.name}_{idx+1}"
            y = _generate_series(length=length, profile=profile, rng=rng)
            if y.size <= horizon + 8:
                continue
            y_tr = y[:-horizon]
            y_te = y[-horizon:]

            L = best_L(y_tr, horizon, per)
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
            # Hybrid neural models (TimesNet / N-BEATS)
            hybrid_models = tuple(m for m in base_models if m != "helformer")
            for model_name in hybrid_models:
                label = label_map[model_name]
                try:
                    model = HybridPlus(
                        base_model_fn=_base_factory(model_name),
                        cfg=cfg,
                        wavelet=wavelet,
                        level=level,
                    ).fit(y_tr)
                    forecasts[label] = model.forecast(y_tr)
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

            # Helformer baseline (no MODWT), trained directly on the original series
            if "helformer" in base_models:
                try:
                    ds_h = WindowDatasetStd(y_tr, L, horizon, stride=1, scale=False)
                    if len(ds_h) > 0:
                        cfg_h = TrainConfig(
                            lookback=L,
                            horizon=horizon,
                            epochs=epochs,
                            batch_size=64,
                            lr=5e-4,
                            weight_decay=1e-4,
                            clip=1.0,
                        )
                        model_h = HelformerAutoRegressor(horizon=horizon, input_dim=1)
                        model_h = train_model(model_h, ds_h, cfg_h)
                        if model_h is not None:
                            xb = torch.from_numpy(y_tr[-L:].astype(np.float32)).view(1, -1, 1)
                            xb = xb.to(cfg_h.device)
                            with torch.no_grad():
                                pred_h = model_h(xb).cpu().numpy().ravel()
                            forecasts["Helformer"] = pred_h
                except Exception as exc:
                    print(f"[{profile.name}:{series_id}] Helformer failed: {exc}")

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
            for name, pred in forecasts.items():
                key = name.replace(" ", "_")
                rec[f"{key}_sMAPE"] = smape(y_te, pred)
                rec[f"{key}_MAPE"] = mape(y_te, pred)
                rec[f"{key}_RMSE"] = rmse(y_te, pred)
                rec[f"{key}_MSE"] = mse(y_te, pred)
            rows.append(rec)

            if plot:
                save_png = out_dir / f"{series_id}.png"
                title = f"{profile.name} {series_id} (H={horizon}, L={L})"
                plot_forecast(title, y_tr, y_te, forecasts, save_path=save_png)

    df = pd.DataFrame(rows)
    metrics_csv = out_dir / "metrics.csv"
    df.to_csv(metrics_csv, index=False)
    print(f"[saved] synthetic metrics: {metrics_csv}")

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


__all__ = ["evaluate_synth_hybrids", "SynthProfile", "PROFILES"]
