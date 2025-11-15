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
)
from ..hybrids import HybridPlus
from ..models import make_model
from ..training import TrainConfig


def _base_factory(name: str):
    def _fn(cfg: TrainConfig):
        return make_model(name, cfg)

    return _fn


def evaluate_m3_hybrids(
    categories: Iterable[str] = ("yearly", "quarterly", "monthly"),
    n_per_cat: int = 8,
    pick: str = "random",
    seed: int = 42,
    epochs: int = 8,
    csv_dir: Path | None = None,
    tsf_dir: Path | None = None,
    out_prefix: Path | None = None,
    wavelet: str = "db4",
    level: int = 1,
    force_rebuild_csv: bool = False,
) -> pd.DataFrame:
    csv_dir = Path(csv_dir or settings.m3_csv_dir)
    tsf_dir = Path(tsf_dir or settings.m3_tsf_dir)
    out_dir = Path(out_prefix or (settings.outputs_dir / "m3_eval"))
    out_dir.parent.mkdir(parents=True, exist_ok=True)

    ensure_m3_csv(csv_dir=csv_dir, tsf_dir=tsf_dir, force_rebuild=force_rebuild_csv)

    rng = np.random.default_rng(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    rows: List[Dict] = []
    out_prefix = str(out_dir)
    for cat in categories:
        H = M3_H[cat]
        per = M3_P[cat]
        pairs = load_train_tsts(cat, csv_dir=csv_dir)
        if not pairs:
            print(f"[{cat}] no pairs found in CSV dir: {csv_dir}")
            continue
        if pick == "first":
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
                lr=3e-3,
                weight_decay=1e-4,
                clip=1.0,
            )
            forecasts: Dict[str, np.ndarray] = {}
            try:
                m_tn = HybridPlus(
                    base_model_fn=_base_factory("timesnet"),
                    cfg=cfg,
                    wavelet=wavelet,
                    level=level,
                ).fit(y_tr)
                forecasts["TimesNet+"] = m_tn.forecast(y_tr)
            except Exception as exc:
                print(f"[{cat}:{sid}] TimesNet+ failed: {exc}")
            try:
                m_nb = HybridPlus(
                    base_model_fn=_base_factory("nbeats"),
                    cfg=cfg,
                    wavelet=wavelet,
                    level=level,
                ).fit(y_tr)
                forecasts["N-BEATS Full"] = m_nb.forecast(y_tr)
            except Exception as exc:
                print(f"[{cat}:{sid}] N-BEATS failed: {exc}")
            if not forecasts:
                naive = seasonal_naive(y_tr, H, per)
                forecasts["TimesNet+"] = naive
                forecasts["N-BEATS Full"] = naive.copy()
            rec = {"category": cat, "series_id": sid}
            for name, pred in forecasts.items():
                rec[f"{name.replace(' ', '_')}_sMAPE"] = smape(y_te, pred)
            rows.append(rec)
            save_png = f"{out_prefix}_{cat}_{sid}.png"
            plot_forecast(f"{cat.upper()} {sid} (H={H}, L={L})", y_tr, y_te, forecasts, save_path=save_png)

    df = pd.DataFrame(rows)
    metrics_csv = f"{out_prefix}_metrics.csv"
    df.to_csv(metrics_csv, index=False)
    print(f"[saved] metrics: {metrics_csv}")
    smape_cols = [c for c in df.columns if c.endswith("_sMAPE")]
    if not df.empty and smape_cols:
        print(df.groupby("category")[smape_cols].mean(numeric_only=True).round(3))
    else:
        print("No results generated — check CSV/logs.")
    return df


__all__ = ["evaluate_m3_hybrids"]
