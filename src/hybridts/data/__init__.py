"""Dataset utilities (M3 processing, loaders, metrics)."""
from .m3 import (
    M3_H,
    M3_P,
    best_L,
    ensure_m3_csv,
    load_train_tsts,
    plot_forecast,
    plot_forecast_test_only,
    seasonal_naive,
    smape,
    mape,
    mse,
    rmse,
    r2_score,
)
from .m4 import M4_H, M4_P, ensure_m4_csv, load_m4_train_test

__all__ = [
    "M3_H",
    "M3_P",
    "M4_H",
    "M4_P",
    "best_L",
    "ensure_m3_csv",
    "ensure_m4_csv",
    "load_train_tsts",
    "load_m4_train_test",
    "plot_forecast",
    "plot_forecast_test_only",
    "seasonal_naive",
    "smape",
    "mape",
    "mse",
    "rmse",
    "r2_score",
]
