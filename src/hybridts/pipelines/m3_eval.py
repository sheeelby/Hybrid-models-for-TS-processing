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
from ..hybrids.decomposition import DecompositionSpec, decompose_series
from ..hybrids import HybridComponent, HybridPlus, build_global_hybrid_components
from ..hybrids import VWHybridMixed
from ..models import (
    DirectNeuralForecaster,
    arima_forecast,
    auto_arima_forecast,
    ets_forecast,
    make_model,
    prophet_forecast,
)
from ..training import TrainConfig
from ..viz import save_component_forecast_plot, save_series_viz_bundle

from ._csv_checkpoints import append_row, reset_csv
from ._model_specs import parse_model_spec

def _progress(iterable, **kwargs):
    if tqdm is None:
        return iterable
    return tqdm(iterable, **kwargs)

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
    stl_kwargs: Mapping[str, Any] | None = None,
    force_rebuild_csv: bool = False,
    force_rebuild_global_components: bool = False,
    series_override: Mapping[str, Sequence[str]] | None = None,
    visualize: bool = False,
    model_params: Mapping[str, Mapping[str, Any]] | None = None,
) -> pd.DataFrame:
    hybrid_models = tuple(parse_model_spec(m) for m in (base_models or ("timesnet", "nbeats")))

    csv_dir = Path(csv_dir or settings.m3_csv_dir)
    tsf_dir = Path(tsf_dir or settings.m3_tsf_dir)
    out_dir = Path(out_prefix or (settings.outputs_dir / "m3_eval"))
    out_dir.mkdir(parents=True, exist_ok=True)

    ensure_m3_csv(csv_dir=csv_dir, tsf_dir=tsf_dir, force_rebuild=force_rebuild_csv)

    metrics_csv = out_dir / "metrics.csv"
    summary_csv = out_dir / "summary.csv"
    reset_csv(metrics_csv)
    reset_csv(summary_csv)

    rng = np.random.default_rng(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    rows: List[Dict] = []
    categories = tuple(categories)
    freq_map = {"yearly": "YE", "quarterly": "QE", "monthly": "ME"}
    model_order = list(dict.fromkeys([spec.label for spec in hybrid_models] + ["ARIMA", "ARIMA_auto", "ETS", "Prophet"]))
    metric_names = ("sMAPE", "MAPE", "RMSE", "MSE")
    metric_cols = [f"{name.replace(' ', '_')}_{metric}" for name in model_order for metric in metric_names]
    series_columns = ["category", "series_id", *metric_cols]
    summary_columns = ["category", "n_series", *metric_cols]
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
        per_eff = per if per and per > 1 else None
        for model_spec in hybrid_models:
            if model_spec.kind != "hybrid_all":
                continue
            decomp_spec = DecompositionSpec(
                method=model_spec.decomposition_method or "modwt",
                wavelet=wavelet,
                level=cat_level,
                boundary=boundary,
                seasonal_period=per_eff,
                stl_kwargs=stl_kwargs,
            )
            params = _effective_model_params(
                model_spec.name,
                base_model_name=model_spec.base_model_name,
                seasonal_period=per_eff,
                model_params=model_params,
            )
            hybrid_ckpt = out_dir / f"{cat}_{model_spec.name}_hybrid_global.pt"
            if hybrid_ckpt.exists() and not force_rebuild_global_components:
                try:
                    ckpt = torch.load(hybrid_ckpt, map_location="cpu")
                    # Backward-compatible checkpoint validation: if metadata keys are
                    # missing (older checkpoints), we accept them; otherwise validate.
                    if "horizon" in ckpt and int(ckpt.get("horizon", -1)) != int(H):
                        raise ValueError("checkpoint params mismatch")
                    if str(ckpt.get("decomposition_method", "modwt")).lower() != decomp_spec.method:
                        raise ValueError("checkpoint params mismatch")
                    if decomp_spec.method == "modwt":
                        if "wavelet" in ckpt and ckpt.get("wavelet") != wavelet:
                            raise ValueError("checkpoint params mismatch")
                        if "level" in ckpt and int(ckpt.get("level", -1)) != int(cat_level):
                            raise ValueError("checkpoint params mismatch")
                        if "boundary" in ckpt and str(ckpt.get("boundary", "wrap")).lower() != str(boundary).lower():
                            raise ValueError("checkpoint params mismatch")
                    else:
                        if "seasonal_period" in ckpt and ckpt.get("seasonal_period") != per_eff:
                            raise ValueError("checkpoint params mismatch")
                        if "stl_kwargs" in ckpt and dict(ckpt.get("stl_kwargs", {})) != dict(stl_kwargs or {}):
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
                            model = _base_factory(model_spec.base_model_name, params=params)(cfg_global)
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
                        global_hybrid_components[model_spec.name] = comps
                        continue
                except Exception as exc:
                    print(f"[{cat}] failed to load hybrid components for {model_spec.name}: {exc}")
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
                base_model_fn=_base_factory(model_spec.base_model_name, params=params),
                wavelet=wavelet,
                level=cat_level,
                boundary=boundary,
                decompose_fn=lambda y_arr, spec=decomp_spec: decompose_series(y_arr, spec=spec, check=True).components,
            )
            global_hybrid_components[model_spec.name] = comps
            try:
                payload = {
                    "category": cat,
                    "model_name": model_spec.name,
                    "horizon": H,
                    "decomposition_method": decomp_spec.method,
                    "wavelet": wavelet,
                    "level": cat_level,
                    "boundary": boundary,
                    "seasonal_period": per_eff,
                    "stl_kwargs": dict(stl_kwargs or {}),
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
                print(f"[{cat}] failed to save hybrid components for {model_spec.name}: {exc}")

        cat_rows: List[Dict] = []
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
            decomposition_specs_for_viz: Dict[str, DecompositionSpec] = {}
            component_forecasts_by_group: Dict[str, Dict[str, Dict[str, np.ndarray]]] = {}
            decomp_cache: dict[tuple[Any, ...], Any] = {}
            for model_spec in hybrid_models:
                label = model_spec.label
                try:
                    per_eff = per if per and per > 1 else None
                    decomp_spec: DecompositionSpec | None = None
                    comps_override: list[np.ndarray] | None = None
                    component_names: tuple[str, ...] | None = None
                    if model_spec.kind != "raw":
                        decomp_spec = DecompositionSpec(
                            method=model_spec.decomposition_method or "modwt",
                            wavelet=wavelet,
                            level=cat_level,
                            boundary=boundary,
                            seasonal_period=per_eff,
                            stl_kwargs=stl_kwargs,
                        )
                        key = decomp_spec.cache_key()
                        if key not in decomp_cache:
                            decomp_cache[key] = decompose_series(y_tr, spec=decomp_spec, check=True)
                        dec_result = decomp_cache[key]
                        comps_override = [np.asarray(comp, float) for comp in dec_result.components]
                        component_names = tuple(dec_result.names)
                        decomposition_specs_for_viz[decomp_spec.group_key] = decomp_spec

                    if model_spec.kind == "hybrid_all":
                        params = _effective_model_params(
                            model_spec.name,
                            base_model_name=model_spec.base_model_name,
                            seasonal_period=per_eff,
                            model_params=model_params,
                        )
                        model = HybridPlus(
                            base_model_fn=_base_factory(model_spec.base_model_name, params=params),
                            cfg=cfg,
                            wavelet=wavelet,
                            level=cat_level,
                            boundary=boundary,
                            pretrained_components=global_hybrid_components.get(model_spec.name),
                            seasonal_period=per_eff,
                            component_names=component_names,
                        ).fit(y_tr, components_override=comps_override)
                        forecasts[label] = model.forecast(y_tr, components_override=comps_override)
                        component_forecasts_by_group.setdefault(decomp_spec.group_key, {})[label] = model.forecast_components(
                            y_tr,
                            components_override=comps_override,
                        )
                    elif model_spec.kind == "raw":
                        params = _effective_model_params(
                            model_spec.name,
                            base_model_name=model_spec.base_model_name,
                            seasonal_period=per_eff,
                            model_params=model_params,
                        )
                        model = DirectNeuralForecaster(
                            base_model_fn=_base_factory(model_spec.base_model_name, params=params),
                            cfg=cfg,
                        ).fit(y_tr)
                        forecasts[label] = model.forecast(y_tr)
                    elif model_spec.kind == "hybrid_mixed":
                        aj_params = _effective_model_params(
                            model_spec.name,
                            base_model_name=model_spec.base_model_name,
                            seasonal_period=per_eff,
                            model_params=model_params,
                        )
                        model = VWHybridMixed(
                            aj_model_fn=_base_factory(model_spec.base_model_name, params=aj_params),
                            neural_component_count=model_spec.neural_component_count,
                            detail_method=str(model_spec.detail_method or "ets"),
                            cfg=cfg,
                            wavelet=wavelet,
                            level=cat_level,
                            boundary=boundary,
                            seasonal_period=per_eff,
                            component_names=component_names,
                        ).fit(y_tr, components_override=comps_override)
                        forecasts[label] = model.forecast(y_tr, components_override=comps_override)
                        component_forecasts_by_group.setdefault(decomp_spec.group_key, {})[label] = model.forecast_components(
                            y_tr,
                            components_override=comps_override,
                        )
                    else:
                        raise ValueError(f"Unknown hybrid model '{model_spec.name}'")
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
                for model_spec in hybrid_models:
                    forecasts[model_spec.label] = naive.copy()
            rec: Dict[str, Any] = {"category": cat, "series_id": sid}
            for col in metric_cols:
                rec[col] = np.nan
            for name, pred in forecasts.items():
                key = name.replace(" ", "_")
                rec[f"{key}_sMAPE"] = smape(y_te, pred)
                rec[f"{key}_MAPE"] = mape(y_te, pred)
                rec[f"{key}_RMSE"] = rmse(y_te, pred)
                rec[f"{key}_MSE"] = mse(y_te, pred)
            rows.append(rec)
            cat_rows.append(rec)
            append_row(metrics_csv, rec, series_columns)
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
                    seasonal_period=per_eff,
                    stl_kwargs=stl_kwargs,
                    decomposition_specs=decomposition_specs_for_viz if decomposition_specs_for_viz else None,
                    component_forecasts_by_group=component_forecasts_by_group if component_forecasts_by_group else None,
                )
            else:
                save_png = out_dir / f"{cat}_{sid}.png"
                plot_forecast(title, y_tr, y_te, forecasts, save_path=save_png)
                if component_forecasts_by_group:
                    multiple_groups = len(component_forecasts_by_group) > 1
                    for group_key, group_forecasts in component_forecasts_by_group.items():
                        spec = decomposition_specs_for_viz.get(group_key)
                        if spec is None:
                            continue
                        suffix = f"_{group_key}" if multiple_groups else ""
                        save_component_forecast_plot(
                            y_tr=y_tr,
                            y_te=y_te,
                            component_forecasts=group_forecasts,
                            decomposition_spec=spec,
                            title=f"{title} component forecasts",
                            save_path=out_dir / f"{cat}_{sid}_components{suffix}.png",
                        )

        if cat_rows:
            cat_df = pd.DataFrame(cat_rows)
            summary_rec: Dict[str, Any] = {"category": cat, "n_series": len(cat_rows)}
            for col in metric_cols:
                summary_rec[col] = float(cat_df[col].mean(skipna=True))
            append_row(summary_csv, summary_rec, summary_columns)

    df = pd.DataFrame(rows)
    print(f"[saved] metrics (per-series): {metrics_csv}")
    print(f"[saved] summary (per-category): {summary_csv}")
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
