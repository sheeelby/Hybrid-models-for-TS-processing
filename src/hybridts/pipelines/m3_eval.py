"""Evaluation helpers for M3 hybrid experiments."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset
try:  # pragma: no cover - optional dependency
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - best-effort fallback
    tqdm = None

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
    mse,
    rmse,
)
from ..hybrids import HybridPlus
from ..models import (
    HelformerAutoRegressor,
    arima_forecast,
    auto_arima_forecast,
    ets_forecast,
    make_model,
    prophet_forecast,
)
from ..training import TrainConfig, WindowDatasetStd, train_model

def _progress(iterable, **kwargs):
    if tqdm is None:
        return iterable
    return tqdm(iterable, **kwargs)

MODEL_LABELS = {
    "timesnet": "TimesNet+",
    "nbeats": "N-BEATS Full",
    "helformer": "Helformer",
}


def _base_factory(name: str):
    def _fn(cfg: TrainConfig):
        return make_model(name, cfg)

    return _fn


def _build_global_helformer_dataset(
    pairs: Sequence[Tuple[str, np.ndarray, np.ndarray]], horizon: int, per: int
) -> Tuple[TensorDataset | None, int]:
    lengths: List[int] = []
    for _, y_tr, _ in pairs:
        lengths.append(best_L(y_tr, horizon, per))
    if not lengths:
        return None, 0
    lookback = max(8, min(lengths))
    xs: List[torch.Tensor] = []
    ys: List[torch.Tensor] = []
    for _, y_tr, _ in pairs:
        ds = WindowDatasetStd(y_tr, lookback, horizon, stride=1, scale=False)
        for idx in range(len(ds)):
            xb, yb = ds[idx]
            xs.append(xb)
            ys.append(yb)
    if not xs:
        return None, lookback
    X = torch.stack(xs)
    Y = torch.stack(ys)
    return TensorDataset(X, Y), lookback


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
    series_override: Mapping[str, Sequence[str]] | None = None,
) -> pd.DataFrame:
    base_models = tuple((m.lower() for m in (base_models or ("timesnet", "nbeats", "helformer"))))
    label_map = {name: MODEL_LABELS.get(name, f"{name.title()}+") for name in base_models}
    use_helformer = "helformer" in base_models
    hybrid_models = tuple(m for m in base_models if m != "helformer")

    csv_dir = Path(csv_dir or settings.m3_csv_dir)
    tsf_dir = Path(tsf_dir or settings.m3_tsf_dir)
    out_dir = Path(out_prefix or (settings.outputs_dir / "m3_eval"))
    out_dir.mkdir(parents=True, exist_ok=True)

    ensure_m3_csv(csv_dir=csv_dir, tsf_dir=tsf_dir, force_rebuild=force_rebuild_csv)

    rng = np.random.default_rng(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    rows: List[Dict] = []
    categories = tuple(categories)
    freq_map = {"yearly": "YE", "quarterly": "QE", "monthly": "ME"}
    for cat in _progress(categories, desc="Categories"):
        H = M3_H[cat]
        per = M3_P[cat]
        pairs = load_train_tsts(cat, csv_dir=csv_dir)
        if not pairs:
            print(f"[{cat}] no pairs found in CSV dir: {csv_dir}")
            continue
        selected_list: List[tuple[str, np.ndarray, np.ndarray]]
        if series_override and cat in series_override:
            wanted = set(series_override[cat])
            selected_list = [triple for triple in pairs if triple[0] in wanted]
            if n_per_cat and n_per_cat > 0:
                selected_list = selected_list[: min(len(selected_list), n_per_cat)]
        elif n_per_cat is None or n_per_cat <= 0:
            selected_list = pairs
        elif pick == "first":
            selected_list = pairs[:n_per_cat]
        elif pick == "last":
            selected_list = pairs[-n_per_cat:]
        else:
            count = min(n_per_cat, len(pairs))
            idx = rng.choice(len(pairs), size=count, replace=False)
            selected_list = [pairs[int(i)] for i in idx]

        global_h_ds: TensorDataset | None = None
        global_model: HelformerAutoRegressor | None = None
        global_cfg: TrainConfig | None = None
        dataset_path = out_dir / f"{cat}_helformer_dataset.pt"
        model_path = out_dir / f"{cat}_helformer_model.pt"
        if use_helformer:
            global_h_ds, global_lookback = _build_global_helformer_dataset(pairs, H, per)
            if global_h_ds is not None and global_lookback > 0:
                global_cfg = TrainConfig(
                    lookback=global_lookback,
                    horizon=H,
                    epochs=max(epochs, 80),
                    batch_size=128,
                    lr=1e-3,
                    weight_decay=1e-4,
                    clip=1.0,
                )
                global_model = HelformerAutoRegressor(
                    horizon=H,
                    input_dim=1,
                    num_heads=4,
                    head_dim=32,
                    lstm_units=32,
                    dropout=0.05,
                    teacher_forcing=0.5,
                )
                # Save dataset to disk for reproducibility
                try:
                    X, Y = global_h_ds.tensors
                    torch.save(
                        {
                            "category": cat,
                            "lookback": global_lookback,
                            "horizon": H,
                            "X": X.cpu(),
                            "Y": Y.cpu(),
                        },
                        dataset_path,
                    )
                except Exception as exc:
                    print(f"[{cat}] failed to save Helformer dataset: {exc}")
                global_model = train_model(global_model, global_h_ds, global_cfg)
                if global_model is not None:
                    try:
                        torch.save(
                            {
                                "category": cat,
                                "state_dict": global_model.state_dict(),
                                "lookback": global_cfg.lookback,
                                "horizon": global_cfg.horizon,
                            },
                            model_path,
                        )
                    except Exception as exc:
                        print(f"[{cat}] failed to save Helformer model: {exc}")

        for sid, y_tr, y_te in _progress(selected_list, desc=f"{cat} series", leave=False):
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
            for model_name in hybrid_models:
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
            # classical baselines
            try:
                forecasts["ARIMA"] = arima_forecast(y_tr, H)
            except Exception as exc:
                print(f"[{cat}:{sid}] ARIMA failed: {exc}")
            try:
                forecasts["ARIMA_auto"] = auto_arima_forecast(y_tr, H)
            except Exception as exc:
                print(f"[{cat}:{sid}] ARIMA_auto failed: {exc}")
            try:
                forecasts["ETS"] = ets_forecast(y_tr, H, seasonal_periods=per)
            except Exception as exc:
                print(f"[{cat}:{sid}] ETS failed: {exc}")
            if use_helformer and global_model is not None and global_cfg is not None:
                try:
                    seq = y_tr.astype(np.float32)
                    if seq.size < global_cfg.lookback:
                        pad = np.full(global_cfg.lookback - seq.size, seq[0], dtype=np.float32)
                        window = np.concatenate([pad, seq])
                    else:
                        window = seq[-global_cfg.lookback :]
                    xb = torch.from_numpy(window).view(1, -1, 1).to(global_cfg.device)
                    with torch.no_grad():
                        pred_h = global_model(xb).cpu().numpy().ravel()
                    forecasts[label_map["helformer"]] = pred_h
                except Exception as exc:
                    print(f"[{cat}:{sid}] Helformer failed: {exc}")
            try:
                freq = freq_map.get(cat, "D")
                forecasts["Prophet"] = prophet_forecast(y_tr, H, freq=freq)
            except Exception as exc:
                print(f"[{cat}:{sid}] Prophet failed: {exc}")
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
                rec[f"{key}_RMSE"] = rmse(y_te, pred)
                rec[f"{key}_MSE"] = mse(y_te, pred)
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
            "RMSE": "_RMSE",
            "MSE": "_MSE",
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
