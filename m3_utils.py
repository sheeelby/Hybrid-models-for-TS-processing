"""Compatibility wrapper around the reorganized M3 utilities."""
from __future__ import annotations

from project_paths import ensure_src_on_path

ensure_src_on_path()

from hybridts.data import (
    M3_H,
    M3_P,
    best_L,
    ensure_m3_csv,
    load_train_tsts,
    plot_forecast,
    seasonal_naive,
    smape,
)

__all__ = [
    "M3_H",
    "M3_P",
    "best_L",
    "ensure_m3_csv",
    "load_train_tsts",
    "plot_forecast",
    "seasonal_naive",
    "smape",
]
