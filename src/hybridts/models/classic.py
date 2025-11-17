"""Классические статистические модели (ARIMA, ETS) для M3."""
from __future__ import annotations

from contextlib import nullcontext
from typing import Iterable, Tuple

import numpy as np

try:  # pragma: no cover - импорт проверяется при запуске пайплайна
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
except Exception:  # pragma: no cover - statsmodels обязателен, но подстрахуемся
    ARIMA = None  # type: ignore[assignment]
    ExponentialSmoothing = None  # type: ignore[assignment]


try:  # pragma: no cover - ACF/PACF �?�+�?�����'��>��?
    from statsmodels.tsa.stattools import acf, pacf
except Exception:  # pragma: no cover
    acf = None  # type: ignore[assignment]
    pacf = None  # type: ignore[assignment]

try:  # pragma: no cover - Prophet �?�+�?�����'��>��?
    import pandas as pd
    from prophet import Prophet

    try:
        from cmdstanpy.utils import disable_logging as _disable_cmdstanpy_logging
    except Exception:  # pragma: no cover - cmdstanpy optional in some envs
        _disable_cmdstanpy_logging = None
except Exception:  # pragma: no cover
    pd = None  # type: ignore[assignment]
    Prophet = None  # type: ignore[assignment]
    _disable_cmdstanpy_logging = None  # type: ignore[assignment]

def _repeat_last(y: np.ndarray, horizon: int) -> np.ndarray:
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if y.size == 0:
        return np.zeros(horizon, dtype=float)
    return np.repeat(float(y[-1]), horizon).astype(float)


def _auto_arima_order(
    data: np.ndarray,
    d: int = 1,
    max_lag: int | None = None,
    max_p: int = 3,
    max_q: int = 3,
) -> Tuple[int, int, int]:
    """Heuristic ARIMA(p,d,q) order selection from ACF/PACF.

    Uses significance bands ~ 1.96/sqrt(N) and picks the largest
    significant lag for PACF -> p, ACF -> q, with small caps.
    Falls back to (1,d,0) on any failure.
    """
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
    order: Tuple[int, int, int] = (1, 1, 0),
) -> np.ndarray:
    """Прогноз ARIMA(p,d,q). Используем statsmodels, fallback — повтор последнего значения."""
    if ARIMA is None:  # pragma: no cover - защита от отсутствующей зависимости
        raise RuntimeError("statsmodels не установлен, ARIMA недоступна")
    data = np.asarray(list(y), dtype=float)
    if data.size < 4:
        return _repeat_last(data, horizon)
    try:
        model = ARIMA(
            data,
            order=order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        fitted = model.fit(method_kwargs={"warn_convergence": False})
        forecast = fitted.forecast(steps=horizon)
        return np.asarray(forecast, dtype=float)
    except Exception:
        return _repeat_last(data, horizon)


def ets_forecast(
    y: Iterable[float],
    horizon: int,
    seasonal_periods: int | None = None,
    trend: str | None = "add",
    seasonal: str | None = "add",
) -> np.ndarray:
    """Прогноз ETS из statsmodels. Если сезонность не задана — строим только тренд."""
    if ExponentialSmoothing is None:  # pragma: no cover
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
    """ARIMA(p,d,q) с автоматическим выбором p и q по ACF/PACF."""
    if ARIMA is None:  # pragma: no cover
        raise RuntimeError("statsmodels не установлен, ARIMA недоступна")
    data = np.asarray(list(y), dtype=float)
    if data.size < 4:
        return _repeat_last(data, horizon)
    try:
        p, d, q = _auto_arima_order(data)
        model = ARIMA(
            data,
            order=(p, d, q),
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        fitted = model.fit(method_kwargs={"warn_convergence": False})
        forecast = fitted.forecast(steps=horizon)
        return np.asarray(forecast, dtype=float)
    except Exception:
        return _repeat_last(data, horizon)


def prophet_forecast(
    y: Iterable[float],
    horizon: int,
    freq: str = "D",
) -> np.ndarray:
    """Baseline Prophet forecast with simple additive seasonality."""
    if Prophet is None or pd is None:  # pragma: no cover
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
