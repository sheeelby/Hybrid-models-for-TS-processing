"""Evaluation helpers for M3 hybrid experiments."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd
import torch

from ..config.settings import settings
from ..data import (
    M3_H,
    M3_P,
    best_L,
    ensure_m3_csv,
    load_train_tsts,
    plot_forecast,
    seasonal_naive,
    smape,
    mape,
    r2_score,
)
from ..hybrids import HybridPlus
from ..models import arima_forecast, ets_forecast, make_model
from ..training import TrainConfig

MODEL_LABELS = {
    "timesnet": "TimesNet+",
    "nbeats": "N-BEATS Full",
    "helformer": "Helformer+",
}


def _base_factory(name: str):
    def _fn(cfg: TrainConfig):
        return make_model(name, cfg)

    return _fn


def evaluate_m3_hybrids(
    categories: Iterable[str] = ("yearly", "quarterly", "monthly"),
    n_per_cat: int | None = None,
    pick: str = "random",
    seed: int = 42,
    epochs: int = 8,
    base_models: Iterable[str] | None = None,
    csv_dir: Path | None = None,
    tsf_dir: Path | None = None,
    out_prefix: Path | None = None,
    wavelet: str = "db4",
    level: int = 1,
    force_rebuild_csv: bool = False,
) -> pd.DataFrame:
    base_models = tuple((m.lower() for m in (base_models or ("timesnet", "nbeats", "helformer"))))
    label_map = {name: MODEL_LABELS.get(name, f"{name.title()}+") for name in base_models}

    csv_dir = Path(csv_dir or settings.m3_csv_dir)
    tsf_dir = Path(tsf_dir or settings.m3_tsf_dir)
    out_dir = Path(out_prefix or (settings.outputs_dir / "m3_eval"))
    out_dir.mkdir(parents=True, exist_ok=True)

    ensure_m3_csv(csv_dir=csv_dir, tsf_dir=tsf_dir, force_rebuild=force_rebuild_csv)

    rng = np.random.default_rng(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    rows: List[Dict] = []
    for cat in categories:
        H = M3_H[cat]
        per = M3_P[cat]
        pairs = load_train_tsts(cat, csv_dir=csv_dir)
        if not pairs:
            print(f"[{cat}] no pairs found in CSV dir: {csv_dir}")
            continue
        if n_per_cat is None or n_per_cat <= 0:
            selected = pairs
        elif pick == "first":
            selected = pairs[:n_per_cat]
        elif pick == "last":
            selected = pairs[-n_per_cat:]
        else:
            count = min(n_per_cat, len(pairs))
            idx = rng.choice(len(pairs), size=count, replace=False)
            selected = [pairs[int(i)] for i in idx]

        for sid, y_tr, y_te in selected:
            L = best_L(y_tr, H, per)
            cfg = TrainConfig(
                lookback=L,
                horizon=H,
                epochs=epochs,
                batch_size=64,
                lr=5e-4,
                weight_decay=1e-4,
                clip=1.0,
            )
            forecasts: Dict[str, np.ndarray] = {}
            for model_name in base_models:
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
                    print(f"[{cat}:{sid}] {label} failed: {exc}")
            # Классические эталонные модели
            try:
                forecasts["ARIMA"] = arima_forecast(y_tr, H)
            except Exception as exc:
                print(f"[{cat}:{sid}] ARIMA failed: {exc}")
            try:
                forecasts["ETS"] = ets_forecast(y_tr, H, seasonal_periods=per)
            except Exception as exc:
                print(f"[{cat}:{sid}] ETS failed: {exc}")
            if not forecasts:
                naive = seasonal_naive(y_tr, H, per)
                for model_name in base_models:
                    label = label_map[model_name]
                    forecasts[label] = naive.copy()
            rec = {"category": cat, "series_id": sid}
            for name, pred in forecasts.items():
                key = name.replace(" ", "_")
                rec[f"{key}_sMAPE"] = smape(y_te, pred)
                rec[f"{key}_MAPE"] = mape(y_te, pred)
                rec[f"{key}_R2"] = r2_score(y_te, pred)
            rows.append(rec)
            save_png = out_dir / f"{cat}_{sid}.png"
            title = f"{cat.upper()} {sid} (H={H}, L={L})"
            plot_forecast(title, y_tr, y_te, forecasts, save_path=save_png)

    df = pd.DataFrame(rows)
    metrics_csv = out_dir / "metrics.csv"
    df.to_csv(metrics_csv, index=False)
    print(f"[saved] metrics: {metrics_csv}")
    if not df.empty:
        metric_suffixes = {
            "sMAPE": "_sMAPE",
            "MAPE": "_MAPE",
            "R2": "_R2",
        }
        for metric, suffix in metric_suffixes.items():
            cols = [c for c in df.columns if c.endswith(suffix)]
            if not cols:
                continue
            print(f"[{metric}] mean by category")
            print(df.groupby("category")[cols].mean(numeric_only=True).round(3))
            overall = df[cols].mean(numeric_only=True)
            print(f"[{metric} overall]")
            print(overall.round(3))
    else:
        print("No results generated — check CSV/logs.")
    return df


__all__ = ["evaluate_m3_hybrids"]
