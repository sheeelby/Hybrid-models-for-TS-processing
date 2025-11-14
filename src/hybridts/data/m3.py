"""Utilities for preparing and loading M3 dataset splits."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

from ..config.settings import settings

M3_H: Dict[str, int] = {"yearly": 6, "quarterly": 8, "monthly": 18}
M3_P: Dict[str, int] = {"yearly": 1, "quarterly": 4, "monthly": 12}


def _csv_paths(csv_dir: Path, cat: str):
    return (
        csv_dir / f"M3_{cat}_TRAIN.csv",
        csv_dir / f"M3_{cat}_TSTS.csv",
    )


def read_tsf(path: Path) -> List[np.ndarray]:
    with open(path, "rb") as fb:
        raw = fb.read()
    for enc in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
        try:
            text = raw.decode(enc)
            break
        except UnicodeDecodeError:
            text = None
    if text is None:
        text = raw.decode("utf-8", errors="ignore")

    rows: List[np.ndarray] = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("@"):
            continue
        parts = line.split(":")
        tail = parts[-1].replace(";", ",")
        vals = [v for v in tail.split(",") if v and v.lower() != "nan"]
        try:
            y = np.array([float(v) for v in vals], dtype=float)
            if y.size > 0:
                rows.append(y)
        except Exception:
            continue
    return rows


def ensure_m3_csv(csv_dir: Path | None = None, tsf_dir: Path | None = None, force_rebuild: bool = False) -> None:
    csv_dir = Path(csv_dir or settings.m3_csv_dir)
    tsf_dir = Path(tsf_dir or settings.m3_tsf_dir)
    csv_dir.mkdir(parents=True, exist_ok=True)
    mapping = {
        "yearly": "m3_yearly_dataset.tsf",
        "quarterly": "m3_quarterly_dataset.tsf",
        "monthly": "m3_monthly_dataset.tsf",
    }
    for cat, tsf_name in mapping.items():
        train_csv, test_csv = _csv_paths(csv_dir, cat)
        have_both = train_csv.exists() and test_csv.exists()
        if have_both and not force_rebuild:
            continue
        tsf_path = tsf_dir / tsf_name
        if not tsf_path.exists():
            print(f"[{cat}] TSF not found at {tsf_path} — skip build")
            continue
        print(f"[{cat}] building CSV from TSF: {tsf_path}")
        series = read_tsf(tsf_path)
        H = M3_H[cat]
        trs, tes = [], []
        for y in series:
            if len(y) > H:
                trs.append(y[:-H])
                tes.append(y[-H:])
        pd.DataFrame({
            "series": [f"T{i+1}" for i in range(len(trs))],
            "values": [",".join(map(str, np.asarray(arr, float))) for arr in trs],
        }).to_csv(train_csv, index=False)
        pd.DataFrame({
            "series": [f"T{i+1}" for i in range(len(tes))],
            "values": [",".join(map(str, np.asarray(arr, float))) for arr in tes],
        }).to_csv(test_csv, index=False)
        print(f"[{cat}] created {train_csv.name}, {test_csv.name} rows={len(trs)}")


def load_train_tsts(cat: str, csv_dir: Path | None = None) -> List[Tuple[str, np.ndarray, np.ndarray]]:
    csv_dir = Path(csv_dir or settings.m3_csv_dir)
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
    data = []
    H = M3_H[cat]
    for sid, s_tr, s_te in zip(tr["series"], tr[val_col_tr], te[val_col_te]):
        try:
            y_tr = np.array([float(v) for v in str(s_tr).replace(";", ",").split(",") if v != ""], dtype=float)
            y_te = np.array([float(v) for v in str(s_te).replace(";", ",").split(",") if v != ""], dtype=float)
        except Exception:
            continue
        if y_tr.size > 4 and y_te.size == H:
            data.append((str(sid), y_tr, y_te))
    return data


def best_L(y_tr: np.ndarray, H: int, per: int, Lmin: int = 16, Lcap: int = 192) -> int:
    L0 = max(2 * H, 3 * per, Lmin)
    L = int(max(Lmin, min(L0, len(y_tr) - H - 4, Lcap)))
    return L


def seasonal_naive(y_tr: np.ndarray, H: int, per: int):
    base = y_tr[-per:]
    reps = int(np.ceil(H / per))
    tiled = np.tile(base, reps)
    return tiled[:H]


def smape(y_true: np.ndarray, y_pred: np.ndarray):
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    denom = np.abs(y_true) + np.abs(y_pred)
    diff = np.abs(y_true - y_pred)
    mask = denom > 0
    return 200 * np.mean(np.divide(diff[mask], denom[mask], out=np.zeros_like(diff[mask]), where=mask))


def plot_forecast(title: str, y_tr: np.ndarray, y_te: np.ndarray, forecasts: Dict[str, np.ndarray], save_path: Path | None = None):
    import matplotlib.pyplot as plt

    H = len(y_te)
    xs_tr = np.arange(len(y_tr))
    xs_te = np.arange(len(y_tr), len(y_tr) + H)
    plt.figure(figsize=(8, 3.2))
    plt.plot(xs_tr, y_tr, label="train")
    plt.plot(xs_te, y_te, label="test")
    for k, v in forecasts.items():
        plt.plot(xs_te, v, label=k)
    plt.title(title)
    plt.xlabel("t")
    plt.legend()
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=120)
    plt.show()
    plt.close()


__all__ = [
    "M3_H",
    "M3_P",
    "ensure_m3_csv",
    "load_train_tsts",
    "best_L",
    "plot_forecast",
    "seasonal_naive",
    "smape",
]
