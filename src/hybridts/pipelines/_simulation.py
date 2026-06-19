"""Shared helpers for full-series simulation modes."""
from __future__ import annotations

import numpy as np


def normalize_simulation_mode(mode: str | None) -> str:
    mode_norm = str(mode or "rollout").strip().lower()
    if mode_norm in {"fitted", "mixed"}:
        return mode_norm
    return "rollout"


def clamp_simulation_mixed_alpha(alpha: float | None, *, default: float = 0.5) -> float:
    try:
        value = float(default if alpha is None else alpha)
    except (TypeError, ValueError):
        value = float(default)
    if not np.isfinite(value):
        value = float(default)
    return float(np.clip(value, 0.0, 1.0))


def simulation_context_window(
    history: np.ndarray,
    out: np.ndarray,
    t: int,
    lookback: int,
    *,
    mode: str,
    mixed_alpha: float = 0.5,
) -> np.ndarray:
    history = np.asarray(history, float).ravel()
    pred_window = np.asarray(out[t - lookback : t], float).copy()
    n = int(history.size)
    mode_norm = normalize_simulation_mode(mode)

    if mode_norm == "fitted" and t < n:
        window = history[max(0, t - lookback) : t]
        if window.size < lookback:
            pad_value = window[0] if window.size else pred_window[0]
            pad = np.repeat(pad_value, lookback - window.size)
            window = np.concatenate([pad, window])
        return window.astype(float, copy=False)

    if mode_norm == "mixed":
        alpha = float(mixed_alpha)
        idx = np.arange(t - lookback, t, dtype=int)
        mask = (idx >= 0) & (idx < n)
        if np.any(mask):
            pred_window[mask] = (
                alpha * history[idx[mask]]
                + (1.0 - alpha) * pred_window[mask]
            )
        return pred_window

    return pred_window


__all__ = [
    "clamp_simulation_mixed_alpha",
    "normalize_simulation_mode",
    "simulation_context_window",
]
