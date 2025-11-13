from __future__ import annotations
import os, glob, random
from typing import Iterable, Dict, List
import numpy as np, pandas as pd, torch

from settings import OUT_DIR, M3_CSV_DIR, M3_TSF_DIR
from m3_utils import (
    M3_H, M3_P,
    ensure_m3_csv, load_train_tsts,
    smape, seasonal_naive, best_L, plot_forecast
)
from train_utils import TrainConfig
from hybrids.modwt_hybrid import HybridPlus
from factories import make_model

def _base_factory(name: str):
    def _fn(cfg: TrainConfig):
        return make_model(name, cfg)
    return _fn

def eval_and_plot_m3_robust(
    categories: Iterable[str] = ("yearly", "quarterly", "monthly"),
    n_per_cat: int = 8,
    pick: str = "random",
    seed: int = 42,
    epochs: int = 8,
    csv_dir: str | os.PathLike = M3_CSV_DIR,   # <-- где лежат CSV
    tsf_dir: str | os.PathLike = M3_TSF_DIR,   # <-- где лежат TSF
    out_prefix: str | None = None,
    wavelet: str = "db4",
    level: int = 1,
    force_rebuild_csv: bool = False
) -> pd.DataFrame:

    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

    # 1) гарантируем CSV: если нет — собираем из TSF
    ensure_m3_csv(csv_dir=csv_dir, tsf_dir=tsf_dir, force_rebuild=force_rebuild_csv)

    # 2) префикс для результатов
    if out_prefix is None:
        out_prefix = os.path.join(OUT_DIR, "m3_ours")

    rows: List[Dict] = []
    for cat in categories:
        H = M3_H[cat]; per = M3_P[cat]
        pairs = load_train_tsts(cat, csv_dir=csv_dir)
        if not pairs:
            print(f"[{cat}] no pairs found in CSV dir: {csv_dir}")
            continue

        if pick == "first":
            selected = pairs[:n_per_cat]
        elif pick == "last":
            selected = pairs[-n_per_cat:]
        else:
            selected = random.sample(pairs, min(n_per_cat, len(pairs)))

        for sid, y_tr, y_te in selected:
            L = best_L(y_tr, H, per)
            cfg = TrainConfig(
                lookback=L, horizon=H, epochs=epochs,
                batch_size=64, lr=3e-3, weight_decay=1e-4, clip=1.0,
                device=('cuda' if torch.cuda.is_available() else 'cpu')
            )

            forecasts: Dict[str, np.ndarray] = {}
            # TimesNet+
            try:
                m_tn = HybridPlus(base_model_fn=_base_factory("timesnet"), cfg=cfg,
                                  wavelet=wavelet, level=level).fit(y_tr)
                forecasts["TimesNet+"] = m_tn.forecast(y_tr)
            except Exception as e:
                print(f"[{cat}:{sid}] TimesNet+ {e}")

            # N-BEATS
            try:
                m_nb = HybridPlus(base_model_fn=_base_factory("nbeats"), cfg=cfg,
                                  wavelet=wavelet, level=level).fit(y_tr)
                forecasts["N-BEATS Full"] = m_nb.forecast(y_tr)
            except Exception as e:
                print(f"[{cat}:{sid}] N-BEATS {e}")

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
        print("Нет результатов — проверь CSV/логи.")

    imgs = glob.glob(f"{out_prefix}_*.png")
    print(f"PNG plots: {len(imgs)}")
    return df