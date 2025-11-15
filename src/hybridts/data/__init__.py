"""Dataset utilities (M3 processing, loaders, metrics)."""
from .m3 import (
    M3_H,
    M3_P,
    best_L,
    ensure_m3_csv,
    load_train_tsts,
    plot_forecast,
    seasonal_naive,
    smape,
    mape,
    mse,
    rmse,
    r2_score,
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
    "mape",
    "mse",
    "rmse",
    "r2_score",
]
