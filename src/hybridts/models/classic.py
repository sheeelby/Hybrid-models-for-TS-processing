from __future__ import annotations

from contextlib import nullcontext
from typing import Iterable, Tuple

import numpy as np

try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
except Exception:  
    ARIMA = None 
    ExponentialSmoothing = None

try:
    from statsmodels.tsa.stattools import acf, pacf
except Exception:
    acf = None
    pacf = None

try:
    import pandas as pd
    from prophet import Prophet

    try:
        from cmdstanpy.utils import disable_logging as _disable_cmdstanpy_logging
    except Exception:
        _disable_cmdstanpy_logging = None
except Exception:
    pd = None
    Prophet = None
    _disable_cmdstanpy_logging = None

def _repeat_last(y: np.ndarray, horizon: int) -> np.ndarray:
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if y.size == 0:
        return np.zeros(horizon, dtype=float)
    return np.repeat(float(y[-1]), horizon).astype(float)


def _clean_series(y: Iterable[float]) -> np.ndarray:
    data = np.asarray(list(y), dtype=float)
    if data.size == 0:
        return data
    mask = np.isfinite(data)
    if not np.all(mask):
        data = data[mask]
    return data.astype(float, copy=False)


def _linear_extrapolation(y: np.ndarray, horizon: int, window: int = 32) -> np.ndarray:
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if y.size == 0:
        return np.zeros(horizon, dtype=float)
    n = int(y.size)
    w = int(min(max(2, window), n))
    t = np.arange(w, dtype=float)
    y_w = y[-w:].astype(float, copy=False)
    y_mean = float(np.mean(y_w))
    y_centered = y_w - y_mean
    slope, intercept = np.polyfit(t, y_centered, deg=1)
    future_t = np.arange(w, w + horizon, dtype=float)
    return (slope * future_t + intercept + y_mean).astype(float)


def _arima_trend_for_d(d: int) -> str:
    return "t" if int(d) > 0 else "c"


def _fit_forecast_arima(data: np.ndarray, horizon: int, order: Tuple[int, int, int]) -> np.ndarray:
    if ARIMA is None:
        raise RuntimeError("statsmodels не установлен, ARIMA недоступна")
    p, d, q = (int(order[0]), int(order[1]), int(order[2]))
    trend = _arima_trend_for_d(d)
    model = ARIMA(
        data,
        order=(p, d, q),
        trend=trend,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    fitted = model.fit(method_kwargs={"warn_convergence": False, "maxiter": 200})
    forecast = fitted.forecast(steps=horizon)
    return np.asarray(forecast, dtype=float)


def _auto_arima_order(
    data: np.ndarray,
    d: int = 1,
    max_lag: int | None = None,
    max_p: int = 3,
    max_q: int = 3,
) -> Tuple[int, int, int]:
    n = int(data.size)
    if n < 10 or acf is None or pacf is None:  # type: ignore[truthy-function]
        return (1, d, 0)
    max_lag = int(max_lag or min(24, n // 2))
    max_lag = max(1, max_lag)
    try:
        acf_vals = acf(data, nlags=max_lag, fft=False)
        pacf_vals = pacf(data, nlags=max_lag, method="ywunbiased")
        crit = 1.96 / np.sqrt(n)
        p_candidates = [lag for lag in range(1, max_lag + 1) if abs(pacf_vals[lag]) > crit]
        q_candidates = [lag for lag in range(1, max_lag + 1) if abs(acf_vals[lag]) > crit]
        p = min(max(p_candidates) if p_candidates else 0, max_p)
        q = min(max(q_candidates) if q_candidates else 0, max_q)
        return (int(p), int(d), int(q))
    except Exception:
        return (1, d, 0)


def arima_forecast(
    y: Iterable[float],
    horizon: int,
    order: Tuple[int, int, int] = (1, 1, 1),
) -> np.ndarray:

    if ARIMA is None:
        raise RuntimeError("statsmodels не установлен, ARIMA недоступна")
    
    data = _clean_series(y)
    if data.size < 6:
        return _linear_extrapolation(data, horizon)
    try:
        forecast = _fit_forecast_arima(data, horizon, order=order)
        if np.nanstd(forecast) < 1e-12 and np.nanstd(np.diff(data[-min(16, data.size) :])) > 1e-12:
            return _linear_extrapolation(data, horizon)
        return forecast
    except Exception:
        for alt in ((0, 1, 1), (1, 1, 0), (0, 1, 0), (1, 0, 1)):
            try:
                return _fit_forecast_arima(data, horizon, order=alt)
            except Exception:
                continue
        return _linear_extrapolation(data, horizon)


def ets_forecast(
    y: Iterable[float],
    horizon: int,
    seasonal_periods: int | None = None,
    trend: str | None = "add",
    seasonal: str | None = "add",
) -> np.ndarray:
    if ExponentialSmoothing is None:
        raise RuntimeError("statsmodels не установлен, ETS недоступна")
    data = np.asarray(list(y), dtype=float)
    if data.size < 4:
        return _repeat_last(data, horizon)
    seasonal_periods = seasonal_periods if seasonal_periods and seasonal_periods > 1 else None
    seasonal = seasonal if seasonal_periods else None
    try:
        model = ExponentialSmoothing(
            data,
            trend=trend,
            seasonal=seasonal,
            seasonal_periods=seasonal_periods,
        )
        fitted = model.fit(optimized=True, use_brute=True)
        forecast = fitted.forecast(horizon)
        return np.asarray(forecast, dtype=float)
    except Exception:
        return _repeat_last(data, horizon)


def auto_arima_forecast(
    y: Iterable[float],
    horizon: int,
) -> np.ndarray:
    if ARIMA is None:
        raise RuntimeError("statsmodels не установлен, ARIMA недоступна")
    data = _clean_series(y)
    if data.size < 6:
        return _linear_extrapolation(data, horizon)
    try:
        p, d, q = _auto_arima_order(data)
        forecast = _fit_forecast_arima(data, horizon, order=(p, d, q))
        if np.nanstd(forecast) < 1e-12 and np.nanstd(np.diff(data[-min(16, data.size) :])) > 1e-12:
            return _linear_extrapolation(data, horizon)
        return forecast
    except Exception:
        best_aic = np.inf
        best_order: Tuple[int, int, int] | None = None
        for d in (0, 1):
            for p in range(0, 3):
                for q in range(0, 3):
                    if p == 0 and q == 0 and d == 0:
                        continue
                    try:
                        model = ARIMA(
                            data,
                            order=(p, d, q),
                            trend=_arima_trend_for_d(d),
                            enforce_stationarity=False,
                            enforce_invertibility=False,
                        )
                        fitted = model.fit(method_kwargs={"warn_convergence": False, "maxiter": 100})
                        aic = float(getattr(fitted, "aic", np.inf))
                        if np.isfinite(aic) and aic < best_aic:
                            best_aic = aic
                            best_order = (p, d, q)
                    except Exception:
                        continue
        if best_order is not None:
            try:
                return _fit_forecast_arima(data, horizon, order=best_order)
            except Exception:
                pass
        return _linear_extrapolation(data, horizon)


def prophet_forecast(
    y: Iterable[float],
    horizon: int,
    freq: str = "D",
) -> np.ndarray:
    if Prophet is None or pd is None:
        raise RuntimeError("prophet не установлен, Prophet недоступен")
    data = np.asarray(list(y), dtype=float)
    if data.size < 2:
        return _repeat_last(data, horizon)
    context = (
        _disable_cmdstanpy_logging()
        if _disable_cmdstanpy_logging is not None
        else nullcontext()
    )
    try:
        start = pd.Timestamp("2000-01-01")
        idx = pd.date_range(start=start, periods=data.size, freq=freq)
        df = pd.DataFrame({"ds": idx, "y": data})
        m = Prophet(
            seasonality_mode="additive",
            yearly_seasonality=False,
            weekly_seasonality=False,
            daily_seasonality=False,
        )
        with context:
            m.fit(df)
            future = m.make_future_dataframe(periods=horizon, freq=freq, include_history=False)
            forecast = m.predict(future)["yhat"].to_numpy()
        return np.asarray(forecast, dtype=float)
    except Exception:
        return _repeat_last(data, horizon)


__all__ = ["arima_forecast", "ets_forecast", "auto_arima_forecast", "prophet_forecast"]