"""Utilities for preparing and loading M4 dataset splits.

The project expects per-category CSVs with two columns:
  - series: series identifier
  - values: comma-separated numeric values
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

from ..config.settings import settings

M4_H: Dict[str, int] = {
    "yearly": 6,
    "quarterly": 8,
    "monthly": 18,
    "weekly": 13,
    "daily": 14,
    "hourly": 48,
}

M4_P: Dict[str, int] = {
    "yearly": 1,
    "quarterly": 4,
    "monthly": 12,
    "weekly": 52,
    "daily": 7,
    "hourly": 24,
}

_CAT_LABEL: Dict[str, str] = {
    "yearly": "Yearly",
    "quarterly": "Quarterly",
    "monthly": "Monthly",
    "weekly": "Weekly",
    "daily": "Daily",
    "hourly": "Hourly",
}


def _csv_paths(csv_dir: Path, cat: str) -> tuple[Path, Path]:
    return (
        csv_dir / f"M4_{cat}_TRAIN.csv",
        csv_dir / f"M4_{cat}_TEST.csv",
    )


def _resolve_raw_dir(raw_dir: Path) -> Path:
    """Resolve where raw M4 wide CSVs live.

    The expected layout is:
      - <raw_dir>/Train/<Freq>-train.csv
      - <raw_dir>/Test/<Freq>-test.csv

    Backward-compatible fallback: if user has a typo folder named `row`, use it.
    """
    raw_dir = Path(raw_dir)
    if raw_dir.exists():
        return raw_dir
    row_dir = raw_dir.with_name("row")
    if row_dir.exists():
        return row_dir
    return raw_dir


def _raw_paths(raw_dir: Path, cat: str) -> tuple[Path, Path]:
    label = _CAT_LABEL[cat]
    train_path = raw_dir / "Train" / f"{label}-train.csv"
    test_path = raw_dir / "Test" / f"{label}-test.csv"
    return train_path, test_path


def _wide_values_to_str(arr: np.ndarray) -> str:
    return ",".join(map(str, np.asarray(arr, float)))


def _normalize_from_raw_wide_csv(
    *,
    cat: str,
    train_raw: Path,
    test_raw: Path,
    out_train: Path,
    out_test: Path,
    horizon: int,
    chunksize: int = 512,
) -> int:
    te_df = pd.read_csv(test_raw, low_memory=False)
    if te_df.empty:
        return 0
    id_col_te = te_df.columns[0]
    te_vals = te_df.drop(columns=[id_col_te]).to_numpy(dtype=float)
    te_ids = te_df[id_col_te].astype(str).to_numpy()
    te_map: Dict[str, np.ndarray] = {}
    for sid, row in zip(te_ids, te_vals):
        arr = np.asarray(row, dtype=float)
        arr = arr[~np.isnan(arr)]
        if arr.size >= horizon:
            te_map[str(sid)] = arr[:horizon]

    out_train.parent.mkdir(parents=True, exist_ok=True)
    out_test.parent.mkdir(parents=True, exist_ok=True)
    n_written = 0
    with open(out_train, "w", newline="", encoding="utf-8") as ftr, open(
        out_test, "w", newline="", encoding="utf-8"
    ) as fte:
        wtr = csv.writer(ftr)
        wte = csv.writer(fte)
        wtr.writerow(["series", "values"])
        wte.writerow(["series", "values"])

        for chunk in pd.read_csv(train_raw, low_memory=False, chunksize=int(chunksize)):
            if chunk.empty:
                continue
            id_col_tr = chunk.columns[0]
            ids = chunk[id_col_tr].astype(str).to_numpy()
            vals = chunk.drop(columns=[id_col_tr]).to_numpy(dtype=float)
            for sid, row in zip(ids, vals):
                y_te = te_map.get(str(sid))
                if y_te is None:
                    continue
                y_tr = np.asarray(row, dtype=float)
                y_tr = y_tr[~np.isnan(y_tr)]
                if y_tr.size <= 1:
                    continue
                wtr.writerow([sid, _wide_values_to_str(y_tr)])
                wte.writerow([sid, _wide_values_to_str(y_te)])
                n_written += 1
    print(f"[m4:{cat}] normalized rows={n_written}")
    return n_written


def ensure_m4_csv(
    csv_dir: Path | None = None,
    raw_dir: Path | None = None,
    categories: Iterable[str] | None = None,
    force_rebuild: bool = False,
) -> None:
    """Ensure normalized M4 CSV files exist for all categories.

    This version only uses local raw wide CSVs from:
      - `raw_dir/Train/<Freq>-train.csv`
      - `raw_dir/Test/<Freq>-test.csv`
    """
    csv_dir = Path(csv_dir or settings.m4_csv_dir)
    raw_dir = _resolve_raw_dir(Path(raw_dir or (settings.data_dir / "m4" / "raw")))
    csv_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    cats = tuple(str(c).lower() for c in (categories or _CAT_LABEL.keys()))
    for cat in cats:
        if cat not in _CAT_LABEL:
            print(f"[m4:{cat}] unknown category; skipping csv build")
            continue
        train_csv, test_csv = _csv_paths(csv_dir, cat)
        have_both = train_csv.exists() and test_csv.exists()
        if have_both and not force_rebuild:
            continue
        train_raw, test_raw = _raw_paths(raw_dir, cat)
        if not train_raw.exists() or not test_raw.exists():
            raise FileNotFoundError(
                f"M4 raw wide CSVs not found for '{cat}'. Expected:\n"
                f"  - {train_raw}\n"
                f"  - {test_raw}\n"
                f"Put files into `src/data/m4/raw/Train` and `src/data/m4/raw/Test` (or pass raw_dir=...)."
            )
        print(f"[m4:{cat}] building normalized CSVs from raw files")
        _normalize_from_raw_wide_csv(
            cat=cat,
            train_raw=train_raw,
            test_raw=test_raw,
            out_train=train_csv,
            out_test=test_csv,
            horizon=M4_H[cat],
        )


def load_m4_train_test(cat: str, csv_dir: Path | None = None) -> List[Tuple[str, np.ndarray, np.ndarray]]:
    csv_dir = Path(csv_dir or settings.m4_csv_dir)
    train_csv, test_csv = _csv_paths(csv_dir, cat)
    if not (train_csv.exists() and test_csv.exists()):
        return []
    tr = pd.read_csv(train_csv, sep=None, engine="python")
    te = pd.read_csv(test_csv, sep=None, engine="python")
    tr.columns = [c.strip().lower() for c in tr.columns]
    te.columns = [c.strip().lower() for c in te.columns]
    if "series" not in tr.columns:
        tr["series"] = [f"T{i+1}" for i in range(len(tr))]
    if "series" not in te.columns:
        te["series"] = [f"T{i+1}" for i in range(len(te))]
    val_col_tr = "values" if "values" in tr.columns else tr.columns[-1]
    val_col_te = "values" if "values" in te.columns else te.columns[-1]
    n = min(len(tr), len(te))
    tr = tr.iloc[:n].reset_index(drop=True)
    te = te.iloc[:n].reset_index(drop=True)
    H = M4_H[cat]
    data = []
    for sid, s_tr, s_te in zip(tr["series"], tr[val_col_tr], te[val_col_te]):
        try:
            y_tr = np.array([float(v) for v in str(s_tr).replace(";", ",").split(",") if v != ""], dtype=float)
            y_te = np.array([float(v) for v in str(s_te).replace(";", ",").split(",") if v != ""], dtype=float)
        except Exception:
            continue
        if y_tr.size > 4 and y_te.size >= H:
            data.append((str(sid), y_tr, y_te[:H]))
    return data


__all__ = ["M4_H", "M4_P", "ensure_m4_csv", "load_m4_train_test"]
