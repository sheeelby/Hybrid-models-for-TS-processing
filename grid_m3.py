from __future__ import annotations

import itertools
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, List

import pandas as pd

from project_paths import ensure_src_on_path

ensure_src_on_path()

from hybridts.config import settings  # noqa: E402
from hybridts.pipelines import evaluate_m3_hybrids  # noqa: E402


@dataclass
class M3Experiment:
    wavelet: str
    level: int
    epochs: int
    seed: int = 42
    n_per_cat: int = 50

    def name(self) -> str:
        return f"w-{self.wavelet}_L-{self.level}_ep-{self.epochs}_s-{self.seed}"


def _run_experiment(exp: M3Experiment) -> pd.DataFrame:
    out_dir = settings.outputs_dir / "m3_grid" / exp.name()
    out_dir.mkdir(parents=True, exist_ok=True)
    df = evaluate_m3_hybrids(
        categories=("yearly", "quarterly", "monthly"),
        n_per_cat=exp.n_per_cat,
        pick="random",
        seed=exp.seed,
        epochs=exp.epochs,
        base_models=("timesnet", "nbeats"),
        out_prefix=out_dir,
        wavelet=exp.wavelet,
        level=exp.level,
        force_rebuild_csv=False,
    )
    df["exp_name"] = exp.name()
    df["wavelet"] = exp.wavelet
    df["level"] = exp.level
    df["epochs"] = exp.epochs
    df["seed"] = exp.seed
    return df


def _summarise(df: pd.DataFrame, exp: M3Experiment) -> pd.DataFrame:
    rows: List[dict] = []
    metric_suffixes = {"sMAPE": "_sMAPE", "MAPE": "_MAPE", "RMSE": "_RMSE", "MSE": "_MSE"}
    for metric, suffix in metric_suffixes.items():
        cols = [c for c in df.columns if c.endswith(suffix)]
        if not cols:
            continue
        cat_means = df.groupby("category")[cols].mean(numeric_only=True)
        overall = df[cols].mean(numeric_only=True)
        for cat, row in cat_means.iterrows():
            rec = {
                "exp_name": exp.name(),
                "wavelet": exp.wavelet,
                "level": exp.level,
                "epochs": exp.epochs,
                "seed": exp.seed,
                "category": cat,
                "metric": metric,
            }
            rec.update(row.to_dict())
            rows.append(rec)
        rec_overall = {
            "exp_name": exp.name(),
            "wavelet": exp.wavelet,
            "level": exp.level,
            "epochs": exp.epochs,
            "seed": exp.seed,
            "category": "ALL",
            "metric": metric,
        }
        rec_overall.update(overall.to_dict())
        rows.append(rec_overall)
    return pd.DataFrame(rows)


def main() -> None:
    wavelets: Iterable[str] = ("db2", "db4", "db6", "sym4")
    levels: Iterable[int] = (1, 2, 3)
    epochs: Iterable[int] = (10, 20)
    seed = 42
    exps = [M3Experiment(w, L, e, seed=seed) for w, L, e in itertools.product(wavelets, levels, epochs)]

    all_metrics: List[pd.DataFrame] = []
    all_summaries: List[pd.DataFrame] = []

    for exp in exps:
        print(f"[run] {exp.name()}")
        df = _run_experiment(exp)
        summary = _summarise(df, exp)
        out_dir = settings.outputs_dir / "m3_grid"
        metrics_path = out_dir / f"{exp.name()}_metrics.csv"
        summary_path = out_dir / f"{exp.name()}_summary.csv"
        df.to_csv(metrics_path, index=False)
        summary.to_csv(summary_path, index=False)
        all_metrics.append(df)
        all_summaries.append(summary)

    if all_summaries:
        combined = pd.concat(all_summaries, ignore_index=True)
        out_dir = settings.outputs_dir / "m3_grid"
        out_dir.mkdir(parents=True, exist_ok=True)
        combined.to_csv(out_dir / "grid_summary.csv", index=False)
        print(f"[saved] global grid summary: {out_dir / 'grid_summary.csv'}")


if __name__ == "__main__":
    main()
