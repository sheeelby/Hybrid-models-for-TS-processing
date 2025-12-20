"""Evaluation helpers for M4 hybrid experiments."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np
import pandas as pd
import torch

try:  # pragma: no cover - optional dependency
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - best-effort fallback
    tqdm = None

from ..config.settings import settings
from ..data import (
    M4_H,
    M4_P,
    best_L,
    ensure_m4_csv,
    load_m4_train_test,
    plot_forecast,
    plot_forecast_test_only,
    seasonal_naive,
    smape,
    mape,
    mse,
    rmse,
)
from ..hybrids import HybridComponent, HybridPlus, VWHybridMixed, build_global_hybrid_components
from ..hybrids.modwt_hybrid import modwt_decompose_with_boundary
from ..models import arima_forecast, auto_arima_forecast, ets_forecast, make_model, prophet_forecast
from ..training import TrainConfig
from ..viz import (
    save_component_forecast_plot,
    save_component_forecast_test_only_plot,
    save_series_viz_bundle,
    save_simulation_full_plot,
    save_simulation_train_plot,
)


def _fitted_one_step_series(
    component: HybridComponent,
    comp_tr: np.ndarray,
    *,
    device: str,
    total_len: int | None = None,
    mode: str = "rollout",
) -> np.ndarray:
    comp_tr = np.asarray(comp_tr, float).ravel()
    n = int(comp_tr.size)
    if n == 0:
        return comp_tr
    if total_len is None:
        total_len = n
    total_len = int(max(1, total_len))
    lookback = int(component.lookback or 0)
    if component.model is None or lookback <= 0 or n <= 2:
        out = np.empty(total_len, dtype=float)
        init_len = min(n, total_len)
        out[:init_len] = comp_tr[:init_len]
        for t in range(1, init_len):
            out[t] = out[t - 1]
        for t in range(init_len, total_len):
            out[t] = out[t - 1]
        return out

    if bool(getattr(component, "per_series_scaling", False)):
        mu = float(np.mean(comp_tr))
        sd = float(np.std(comp_tr) + 1e-8)
    else:
        mu = float(component.mu)
        sd = float(component.sd + 1e-8)

    lookback = min(lookback, max(1, n - 1))
    out = np.empty(total_len, dtype=float)
    init_len = min(lookback, n, total_len)
    out[:init_len] = comp_tr[:init_len]
    if init_len < lookback:
        # pad seed if series shorter than lookback
        for t in range(init_len, min(lookback, total_len)):
            out[t] = out[t - 1]
        init_len = min(lookback, total_len)
    model = component.model
    model.eval()
    mode = str(mode or "rollout").lower()
    for t in range(init_len, total_len):
        if mode == "fitted" and t < n:
            window = comp_tr[max(0, t - lookback) : t]
            if window.size < lookback:
                pad = np.repeat(window[0] if window.size else out[0], lookback - window.size)
                window = np.concatenate([pad, window])
        else:
            window = out[t - lookback : t]
        xb = ((window - mu) / sd).astype(np.float32).reshape(1, 1, -1)
        with torch.no_grad():
            pred = model(torch.from_numpy(xb).to(device)).detach().cpu().numpy().ravel()
        if pred.size <= 0:
            out[t] = out[t - 1]
        else:
            out[t] = float(pred[0]) * sd + mu
    return out


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


def _cat_level(cat: str, base_level: int) -> int:
    base_level = int(base_level)
    suggested = {
        "yearly": base_level,
        "quarterly": max(base_level, 2),
        "monthly": max(base_level, 3),
        "weekly": max(base_level, 3),
        "daily": max(base_level, 4),
        "hourly": max(base_level, 5),
    }
    return int(suggested.get(str(cat).lower(), base_level))


def evaluate_m4_hybrids(
    categories: Iterable[str] = ("yearly", "quarterly", "monthly", "weekly", "daily", "hourly"),
    n_per_cat: int | None = None,
    pick: str = "random",
    seed: int = 42,
    epochs: int = 8,
    base_models: Iterable[str] | None = None,
    csv_dir: Path | None = None,
    raw_dir: Path | None = None,
    out_prefix: Path | None = None,
    wavelet: str = "db4",
    level: int = 1,
    boundary: str = "wrap",
    force_rebuild_csv: bool = False,
    force_rebuild_global_components: bool = False,
    series_override: Mapping[str, Sequence[str]] | None = None,
    visualize: bool = False,
    plot_test_only: bool = True,
    use_full_modwt_components: bool = True,
    simulate_full_series: bool = False,
    simulation_mode: str = "rollout",
    simulation_train_only_plot: bool = False,
    model_params: Mapping[str, Mapping[str, Any]] | None = None,
) -> pd.DataFrame:
    base_models = tuple((m.lower() for m in (base_models or ("timesnet", "nbeats"))))
    label_map = {name: MODEL_LABELS.get(name, f"{name.title()}+") for name in base_models}
    hybrid_models = base_models

    csv_dir = Path(csv_dir or settings.m4_csv_dir)
    out_dir = Path(out_prefix or (settings.outputs_dir / "m4_eval"))
    out_dir.mkdir(parents=True, exist_ok=True)

    ensure_m4_csv(
        csv_dir=csv_dir,
        raw_dir=raw_dir,
        categories=categories,
        force_rebuild=force_rebuild_csv,
    )

    rng = np.random.default_rng(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    rows: List[Dict] = []
    categories = tuple(categories)
    freq_map = {
        "yearly": "YE",
        "quarterly": "QE",
        "monthly": "ME",
        "weekly": "W",
        "daily": "D",
        "hourly": "H",
    }
    for cat in _progress(categories, desc="Categories"):
        cat = str(cat).lower()
        if cat not in M4_H:
            print(f"[m4:{cat}] unknown category; skipping")
            continue
        H = M4_H[cat]
        per = M4_P[cat]
        cat_level = _cat_level(cat, level)
        pairs = load_m4_train_test(cat, csv_dir=csv_dir)
        if not pairs:
            print(f"[m4:{cat}] no pairs found in CSV dir: {csv_dir}")
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
        # For M4 we optionally use MODWT components computed on the full series
        # (train+test) to avoid boundary mismatch; in that mode, global pretraining
        # is disabled to keep training consistent.
        global_hybrid_components: Dict[str, List[HybridComponent]] = {}
        if not use_full_modwt_components:
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
                        if "horizon" in ckpt and int(ckpt.get("horizon", -1)) != int(H):
                            raise ValueError("checkpoint params mismatch")
                        if "wavelet" in ckpt and ckpt.get("wavelet") != wavelet:
                            raise ValueError("checkpoint params mismatch")
                        if "level" in ckpt and int(ckpt.get("level", -1)) != int(cat_level):
                            raise ValueError("checkpoint params mismatch")
                        if "boundary" in ckpt and str(ckpt.get("boundary", "wrap")).lower() != str(boundary).lower():
                            raise ValueError("checkpoint params mismatch")
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
                        print(f"[m4:{cat}] failed to load hybrid components for {model_name}: {exc}")

                hybrid_cfg = TrainConfig(
                    lookback=max(16, min(256, max(32, 2 * H, 3 * per))),
                    horizon=H,
                    epochs=max(int(epochs), 2),
                    batch_size=128,
                    lr=3e-4,
                    weight_decay=1e-4,
                    clip=1.0,
                )
                comps = build_global_hybrid_components(
                    selected_list,
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
                    print(f"[m4:{cat}] failed to save hybrid components for {model_name}: {exc}")

        for sid, y_tr, y_te in _progress(selected_list, desc=f"{cat} series", leave=False):
            L = best_L(y_tr, H, per)
            comps_override: list[np.ndarray] | None = None
            comps_full: list[np.ndarray] | None = None
            if use_full_modwt_components:
                y_full = np.concatenate([np.asarray(y_tr, float), np.asarray(y_te, float)], axis=0)
                A_full, D_full = modwt_decompose_with_boundary(
                    y_full, wavelet=wavelet, level=cat_level, boundary=boundary, check=True
                )
                comps_full = [A_full] + D_full if len(D_full) else [A_full]
                comps_override = [np.asarray(c[: len(y_tr)], float) for c in comps_full]
            total_len = int(len(y_tr) + H)
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
            simulations: Dict[str, np.ndarray] = {}
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
                            pretrained_components=None if use_full_modwt_components else global_hybrid_components.get(model_name),
                            seasonal_period=per_eff,
                        ).fit(y_tr, components_override=comps_override)
                        forecasts[label] = model.forecast(y_tr, components_override=comps_override)
                        component_forecasts[label] = model.forecast_components(y_tr, components_override=comps_override)
                        if simulate_full_series and comps_override is not None and comps_full is not None:
                            fitted_components = []
                            for comp_obj, comp_full_arr in zip(model.components, comps_full):
                                comp_tr_arr = np.asarray(comp_full_arr[: len(y_tr)], float)
                                fitted_components.append(
                                    _fitted_one_step_series(
                                        comp_obj,
                                        comp_tr_arr,
                                        device=cfg.device,
                                        total_len=total_len,
                                        mode=simulation_mode,
                                    )
                                )
                            y_sim = np.sum(np.stack(fitted_components, 0), axis=0)
                            # If we simulated beyond train+H (shouldn't), trim.
                            simulations[label] = np.asarray(y_sim[:total_len], float)
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
                        ).fit(y_tr, components_override=comps_override)
                        forecasts[label] = model.forecast(y_tr, components_override=comps_override)
                        component_forecasts[label] = model.forecast_components(y_tr, components_override=comps_override)
                        if simulate_full_series and comps_full is not None and model.aj_component is not None:
                            A_tr_arr = np.asarray(comps_full[0][: len(y_tr)], float)
                            Aj_sim = _fitted_one_step_series(
                                model.aj_component,
                                A_tr_arr,
                                device=cfg.device,
                                total_len=total_len,
                                mode=simulation_mode,
                            )
                            details_sum = np.zeros(total_len, dtype=float)
                            # Use true details on train, predicted details on test (H).
                            comp_map = component_forecasts.get(label, {})
                            for j, dj_full in enumerate(comps_full[1:], start=1):
                                dj_tr = np.asarray(dj_full[: len(y_tr)], float)
                                dj_pred = np.asarray(comp_map.get(f"D_{j}", np.zeros(H)), float).ravel()
                                if dj_pred.size != H:
                                    dj_pred = np.pad(dj_pred, (0, max(0, H - dj_pred.size)), mode="edge")[:H]
                                dj_series = np.concatenate([dj_tr, dj_pred], axis=0)
                                if dj_series.size < total_len:
                                    dj_series = np.pad(dj_series, (0, total_len - dj_series.size), mode="edge")
                                details_sum += dj_series[:total_len]
                            simulations[label] = (Aj_sim + details_sum)[:total_len]
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
                        ).fit(y_tr, components_override=comps_override)
                        forecasts[label] = model.forecast(y_tr, components_override=comps_override)
                        component_forecasts[label] = model.forecast_components(y_tr, components_override=comps_override)
                        if simulate_full_series and comps_full is not None and model.aj_component is not None:
                            A_tr_arr = np.asarray(comps_full[0][: len(y_tr)], float)
                            Aj_sim = _fitted_one_step_series(
                                model.aj_component,
                                A_tr_arr,
                                device=cfg.device,
                                total_len=total_len,
                                mode=simulation_mode,
                            )
                            details_sum = np.zeros(total_len, dtype=float)
                            comp_map = component_forecasts.get(label, {})
                            for j, dj_full in enumerate(comps_full[1:], start=1):
                                dj_tr = np.asarray(dj_full[: len(y_tr)], float)
                                dj_pred = np.asarray(comp_map.get(f"D_{j}", np.zeros(H)), float).ravel()
                                if dj_pred.size != H:
                                    dj_pred = np.pad(dj_pred, (0, max(0, H - dj_pred.size)), mode="edge")[:H]
                                dj_series = np.concatenate([dj_tr, dj_pred], axis=0)
                                if dj_series.size < total_len:
                                    dj_series = np.pad(dj_series, (0, total_len - dj_series.size), mode="edge")
                                details_sum += dj_series[:total_len]
                            simulations[label] = (Aj_sim + details_sum)[:total_len]
                    else:
                        raise ValueError(f"Unknown hybrid model '{model_name}'")
                except Exception as exc:
                    print(f"[m4:{cat}:{sid}] {label} failed: {exc}")

            # Classical baselines
            try:
                forecasts["ARIMA"] = arima_forecast(y_tr, H)
            except Exception as exc:
                print(f"[m4:{cat}:{sid}] ARIMA failed: {exc}")
            try:
                forecasts["ARIMA_auto"] = auto_arima_forecast(y_tr, H)
            except Exception as exc:
                print(f"[m4:{cat}:{sid}] ARIMA_auto failed: {exc}")
            try:
                forecasts["ETS"] = ets_forecast(y_tr, H, seasonal_periods=per)
            except Exception as exc:
                print(f"[m4:{cat}:{sid}] ETS failed: {exc}")
            try:
                freq = freq_map.get(cat, "D")
                forecasts["Prophet"] = prophet_forecast(y_tr, H, freq=freq)
            except Exception as exc:
                print(f"[m4:{cat}:{sid}] Prophet failed: {exc}")

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
                if simulate_full_series and simulations:
                    if simulation_train_only_plot:
                        save_simulation_train_plot(
                            y_tr=np.asarray(y_tr, float),
                            simulations=simulations,
                            title=f"{title} simulation",
                            save_path=(out_dir / "viz" / "09_simulation_train" / f"{series_key}.png"),
                        )
                    else:
                        save_simulation_full_plot(
                            y_tr=np.asarray(y_tr, float),
                            y_te=np.asarray(y_te, float),
                            simulations=simulations,
                            title=f"{title} simulation (full series)",
                            save_path=(out_dir / "viz" / "09_simulation_full" / f"{series_key}.png"),
                        )
            else:
                save_png = out_dir / f"{cat}_{sid}.png"
                if plot_test_only:
                    save_png = out_dir / f"{cat}_{sid}_test_only.png"
                    plot_forecast_test_only(title, y_te, forecasts, save_path=save_png)
                else:
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
                    save_component_forecast_test_only_plot(
                        y_tr=y_tr,
                        y_te=y_te,
                        component_forecasts=component_forecasts,
                        wavelet=wavelet,
                        level=cat_level,
                        boundary=boundary,
                        title=f"{title} component forecasts (test only)",
                        save_path=out_dir / f"{cat}_{sid}_components_test_only.png",
                    )
                if simulate_full_series and simulations:
                    if simulation_train_only_plot:
                        save_simulation_train_plot(
                            y_tr=np.asarray(y_tr, float),
                            simulations=simulations,
                            title=f"{title} simulation (train only)",
                            save_path=out_dir / f"{cat}_{sid}_simulation_train.png",
                        )
                    else:
                        save_simulation_full_plot(
                            y_tr=np.asarray(y_tr, float),
                            y_te=np.asarray(y_te, float),
                            simulations=simulations,
                            title=f"{title} simulation (full series)",
                            save_path=out_dir / f"{cat}_{sid}_simulation_full.png",
                        )

    df = pd.DataFrame(rows)
    metrics_csv = out_dir / "metrics.csv"
    df.to_csv(metrics_csv, index=False)
    print(f"[saved] metrics: {metrics_csv}")
    if df.empty:
        print("No results generated — check CSV/logs.")
        return df

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
    return df


__all__ = ["evaluate_m4_hybrids"]
