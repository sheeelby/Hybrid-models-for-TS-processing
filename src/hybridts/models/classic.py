"""Классические статистические модели (ARIMA, ETS) для M3."""
from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np

try:  # pragma: no cover - импорт проверяется при запуске пайплайна
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
except Exception:  # pragma: no cover - statsmodels обязателен, но подстрахуемся
    ARIMA = None  # type: ignore[assignment]
    ExponentialSmoothing = None  # type: ignore[assignment]


def _repeat_last(y: np.ndarray, horizon: int) -> np.ndarray:
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if y.size == 0:
        return np.zeros(horizon, dtype=float)
    return np.repeat(float(y[-1]), horizon).astype(float)


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


__all__ = ["arima_forecast", "ets_forecast"]
