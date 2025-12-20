"""Evaluation helpers for M3 hybrid experiments."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
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
from ..hybrids import HybridComponent, HybridPlus, build_global_hybrid_components
from ..hybrids import VWHybridMixed
from ..models import (
    arima_forecast,
    auto_arima_forecast,
    ets_forecast,
    make_model,
    prophet_forecast,
)
from ..training import TrainConfig
from ..viz import save_component_forecast_plot, save_series_viz_bundle

def _progress(iterable, **kwargs):
    if tqdm is None:
        return iterable
    return tqdm(iterable, **kwargs)

MODEL_LABELS = {
    "timesnet": "TimesNet+",
    "nbeats": "N-BEATS Full",
    "vw_timesnet_ets": "VW + TimesNet + ETS",
    "vw_timesnet_arima_auto": "VW + TimesNet + Arima_auto",
    "vw_nbeats_ets": "VW + N-Beats + ETS",
    "vw_nbeats_arima_auto": "VW + N-Beats + Arima_auto",
}


def _base_factory(name: str, params: Mapping[str, Any] | None = None):
    def _fn(cfg: TrainConfig):
        return make_model(name, cfg, params=params)

    return _fn


def _effective_model_params(
    model_name: str,
    *,
    base_model_name: str | None = None,
    seasonal_period: int | None,
    model_params: Mapping[str, Mapping[str, Any]] | None,
) -> Mapping[str, Any] | None:
    base_name = (base_model_name or model_name).lower()
    primary = model_name.lower()
    params_raw = None
    if model_params:
        params_raw = model_params.get(primary)
        if params_raw is None and base_name != primary:
            params_raw = model_params.get(base_name)
    params = dict(params_raw) if params_raw else None
    if base_name == "nbeats" and (seasonal_period is None or seasonal_period <= 1):
        if params is None:
            params = {}
        params.setdefault("use_seasonality", False)
    return params


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
    boundary: str = "wrap",
    force_rebuild_csv: bool = False,
    force_rebuild_global_components: bool = False,
    series_override: Mapping[str, Sequence[str]] | None = None,
    visualize: bool = False,
    model_params: Mapping[str, Mapping[str, Any]] | None = None,
) -> pd.DataFrame:
    base_models = tuple((m.lower() for m in (base_models or ("timesnet", "nbeats"))))
    label_map = {name: MODEL_LABELS.get(name, f"{name.title()}+") for name in base_models}
    hybrid_models = base_models

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
        cat_level = 3 if cat == "monthly" else 2 if cat == "quarterly" else level
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

        # Global hybrid components for neural base models (TimesNet / N-BEATS).
        global_hybrid_components: Dict[str, List[HybridComponent]] = {}
        for model_name in hybrid_models:
            if model_name not in {"timesnet", "nbeats"}:
                continue
            params = _effective_model_params(
                model_name,
                seasonal_period=(per if per and per > 1 else None),
                model_params=model_params,
            )
            hybrid_ckpt = out_dir / f"{cat}_{model_name}_hybrid_global.pt"
            if hybrid_ckpt.exists() and not force_rebuild_global_components:
                try:
                    ckpt = torch.load(hybrid_ckpt, map_location="cpu")
                    # Backward-compatible checkpoint validation: if metadata keys are
                    # missing (older checkpoints), we accept them; otherwise validate.
                    if "horizon" in ckpt and int(ckpt.get("horizon", -1)) != int(H):
                        raise ValueError("checkpoint params mismatch")
                    if "wavelet" in ckpt and ckpt.get("wavelet") != wavelet:
                        raise ValueError("checkpoint params mismatch")
                    if "level" in ckpt and int(ckpt.get("level", -1)) != int(cat_level):
                        raise ValueError("checkpoint params mismatch")
                    if "boundary" in ckpt and str(ckpt.get("boundary", "wrap")).lower() != str(boundary).lower():
                        raise ValueError("checkpoint params mismatch")
                    # Old checkpoints (before per-series scaling) often produce large
                    # level shifts ("downward drift") on M3. Rebuild them automatically.
                    comps_meta = list(ckpt.get("components", []))
                    if any(("per_series_scaling" not in item) for item in comps_meta):
                        raise ValueError("checkpoint too old (missing per_series_scaling)")
                    if any(not bool(item.get("per_series_scaling", False)) for item in comps_meta):
                        raise ValueError("checkpoint too old (global scaling)")
                    comps: List[HybridComponent] = []
                    for item in ckpt.get("components", []):
                        state_dict = item.get("state_dict")
                        lookback = item.get("lookback")
                        mu = float(item.get("mu", 0.0))
                        sd = float(item.get("sd", 1.0))
                        model = None
                        if state_dict is not None and lookback is not None:
                            cfg_global = TrainConfig(
                                lookback=int(lookback),
                                horizon=H,
                                epochs=0,
                                batch_size=128,
                                lr=1e-3,
                                weight_decay=1e-4,
                                clip=1.0,
                            )
                            model = _base_factory(model_name, params=params)(cfg_global)
                            model.load_state_dict(state_dict)
                            model.to(cfg_global.device)
                            model.eval()
                        comps.append(
                            HybridComponent(
                                model=model,
                                mu=mu,
                                sd=sd,
                                lookback=lookback,
                                per_series_scaling=bool(item.get("per_series_scaling", True)),
                            )
                        )
                    if comps:
                        global_hybrid_components[model_name] = comps
                        continue
                except Exception as exc:
                    print(f"[{cat}] failed to load hybrid components for {model_name}: {exc}")
            # No usable checkpoint -> build global components.
            try:
                global_L = max(8, min(best_L(y_tr, H, per) for _, y_tr, _ in pairs))
            except ValueError:
                global_L = 8
            hybrid_cfg = TrainConfig(
                lookback=global_L,
                horizon=H,
                epochs=max(epochs, 2),
                batch_size=128,
                lr=3e-4,
                weight_decay=1e-4,
                clip=1.0,
            )
            comps = build_global_hybrid_components(
                pairs,
                hybrid_cfg,
                base_model_fn=_base_factory(model_name, params=params),
                wavelet=wavelet,
                level=cat_level,
                boundary=boundary,
            )
            global_hybrid_components[model_name] = comps
            try:
                payload = {
                    "category": cat,
                    "model_name": model_name,
                    "horizon": H,
                    "wavelet": wavelet,
                    "level": cat_level,
                    "boundary": boundary,
                    "components": [],
                }
                for comp in comps:
                    state = comp.model.state_dict() if comp.model is not None else None
                    payload["components"].append(
                        {
                            "state_dict": state,
                            "mu": comp.mu,
                            "sd": comp.sd,
                            "lookback": comp.lookback,
                            "per_series_scaling": bool(getattr(comp, "per_series_scaling", False)),
                        }
                    )
                torch.save(payload, hybrid_ckpt)
            except Exception as exc:
                print(f"[{cat}] failed to save hybrid components for {model_name}: {exc}")
        for sid, y_tr, y_te in _progress(selected_list, desc=f"{cat} series", leave=False):
            L = best_L(y_tr, H, per)
            cfg = TrainConfig(
                lookback=L,
                horizon=H,
                epochs=epochs,
                batch_size=64,
                lr=3e-4,
                weight_decay=2e-4,
                clip=0.5,
            )
            forecasts: Dict[str, np.ndarray] = {}
            component_forecasts: Dict[str, Dict[str, np.ndarray]] = {}
            for model_name in hybrid_models:
                label = label_map[model_name]
                try:
                    per_eff = per if per and per > 1 else None
                    if model_name in {"timesnet", "nbeats"}:
                        params = _effective_model_params(
                            model_name,
                            seasonal_period=per_eff,
                            model_params=model_params,
                        )
                        model = HybridPlus(
                            base_model_fn=_base_factory(model_name, params=params),
                            cfg=cfg,
                            wavelet=wavelet,
                            level=cat_level,
                            boundary=boundary,
                            pretrained_components=global_hybrid_components.get(model_name),
                            seasonal_period=per_eff,
                        ).fit(y_tr)
                        forecasts[label] = model.forecast(y_tr)
                        component_forecasts[label] = model.forecast_components(y_tr)
                    elif model_name in {"vw_timesnet_ets", "vw_timesnet_arima_auto"}:
                        detail = "ets" if model_name.endswith("_ets") else "arima_auto"
                        aj_params = _effective_model_params(
                            model_name,
                            base_model_name="timesnet",
                            seasonal_period=per_eff,
                            model_params=model_params,
                        )
                        model = VWHybridMixed(
                            aj_model_fn=_base_factory("timesnet", params=aj_params),
                            detail_method=detail,
                            cfg=cfg,
                            wavelet=wavelet,
                            level=cat_level,
                            seasonal_period=per_eff,
                        ).fit(y_tr)
                        forecasts[label] = model.forecast(y_tr)
                        component_forecasts[label] = model.forecast_components(y_tr)
                    elif model_name in {"vw_nbeats_ets", "vw_nbeats_arima_auto"}:
                        detail = "ets" if model_name.endswith("_ets") else "arima_auto"
                        aj_params = _effective_model_params(
                            model_name,
                            base_model_name="nbeats",
                            seasonal_period=per_eff,
                            model_params=model_params,
                        )
                        model = VWHybridMixed(
                            aj_model_fn=_base_factory("nbeats", params=aj_params),
                            detail_method=detail,
                            cfg=cfg,
                            wavelet=wavelet,
                            level=cat_level,
                            seasonal_period=per_eff,
                        ).fit(y_tr)
                        forecasts[label] = model.forecast(y_tr)
                        component_forecasts[label] = model.forecast_components(y_tr)
                    else:
                        raise ValueError(f"Unknown hybrid model '{model_name}'")
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
            title = f"{cat.upper()} {sid} (H={H}, L={L})"
            if visualize:
                series_key = f"{cat}_{sid}"
                save_series_viz_bundle(
                    out_dir=out_dir / "viz",
                    series_key=series_key,
                    title_prefix=title,
                    y_tr=y_tr,
                    y_te=y_te,
                    forecasts=forecasts,
                    wavelet=wavelet,
                    level=cat_level,
                    boundary=boundary,
                    component_forecasts=component_forecasts if component_forecasts else None,
                )
            else:
                save_png = out_dir / f"{cat}_{sid}.png"
                plot_forecast(title, y_tr, y_te, forecasts, save_path=save_png)
                if component_forecasts:
                    save_component_forecast_plot(
                        y_tr=y_tr,
                        y_te=y_te,
                        component_forecasts=component_forecasts,
                        wavelet=wavelet,
                        level=cat_level,
                        boundary=boundary,
                        title=f"{title} component forecasts",
                        save_path=out_dir / f"{cat}_{sid}_components.png",
                    )

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
