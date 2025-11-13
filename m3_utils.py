from __future__ import annotations
import os
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from settings import M3_CSV_DIR, M3_TSF_DIR

M3_H: Dict[str, int] = {"yearly": 6, "quarterly": 8, "monthly": 18}
M3_P: Dict[str, int] = {"yearly": 1, "quarterly": 4, "monthly": 12}

# --- робастный TSF reader (см. предыдущий патч) ---
def read_tsf(path: str) -> List[np.ndarray]:
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
        vals = [v for v in tail.split(",") if v != "" and v.lower() != "nan"]
        try:
            y = np.array([float(v) for v in vals], dtype=float)
            if y.size > 0:
                rows.append(y)
        except Exception:
            pass
    return rows

def _csv_paths(csv_dir, cat):
    csv_dir = os.fspath(csv_dir)
    return (
        os.path.join(csv_dir, f"M3_{cat}_TRAIN.csv"),
        os.path.join(csv_dir, f"M3_{cat}_TSTS.csv"),
    )

def ensure_m3_csv(csv_dir=M3_CSV_DIR, tsf_dir=M3_TSF_DIR, force_rebuild: bool=False) -> None:
    """
    Гарантирует наличие M3_*_{TRAIN|TSTS}.csv в csv_dir.
    Если нет — собирает из .tsf в tsf_dir. Если force_rebuild=True — пересобирает всегда.
    """
    os.makedirs(csv_dir, exist_ok=True)
    mapping = {
        "yearly":    "m3_yearly_dataset.tsf",
        "quarterly": "m3_quarterly_dataset.tsf",
        "monthly":   "m3_monthly_dataset.tsf",
    }
    for cat, tsf_name in mapping.items():
        train_csv, test_csv = _csv_paths(csv_dir, cat)
        have_both = os.path.exists(train_csv) and os.path.exists(test_csv)
        if have_both and not force_rebuild:
            continue
        tsf_path = os.path.join(tsf_dir, tsf_name)
        if not os.path.exists(tsf_path):
            print(f"[{cat}] TSF not found at {tsf_path} — пропуск сборки.")
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
            "values": [",".join(map(str, np.asarray(arr, float))) for arr in trs]
        }).to_csv(train_csv, index=False)

        pd.DataFrame({
            "series": [f"T{i+1}" for i in range(len(tes))],
            "values": [",".join(map(str, np.asarray(arr, float))) for arr in tes]
        }).to_csv(test_csv, index=False)

        print(f"[{cat}] created {os.path.basename(train_csv)}, {os.path.basename(test_csv)} rows={len(trs)}")

def load_train_tsts(cat: str, csv_dir=M3_CSV_DIR) -> List[Tuple[str, np.ndarray, np.ndarray]]:
    """
    Загружает пары (series_id, y_train, y_test) из CSV каталога csv_dir.
    """
    import numpy as np
    import pandas as pd

    train_csv, test_csv = _csv_paths(csv_dir, cat)
    if not (os.path.exists(train_csv) and os.path.exists(test_csv)):
        return []

    tr = pd.read_csv(train_csv, sep=None, engine="python")
    te = pd.read_csv(test_csv,  sep=None, engine="python")
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

# ---------- Эвристика lookback ----------

def best_L(y_tr: np.ndarray, H: int, per: int, Lmin: int = 16, Lcap: int = 192) -> int:
    L0 = max(2 * H, 3 * per, Lmin)
    L = int(max(Lmin, min(L0, len(y_tr) - H - 4, Lcap)))
    return L

# ---------- Визуализация ----------

def plot_forecast(title: str, y_tr: np.ndarray, y_te: np.ndarray, forecasts: Dict[str, np.ndarray], save_path: str | None = None):
    H = len(y_te)
    xs_tr = np.arange(len(y_tr))
    xs_te = np.arange(len(y_tr), len(y_tr) + H)
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 3.2))
    plt.plot(xs_tr, y_tr, label="train")
    plt.plot(xs_te, y_te, label="test")
    for k, v in forecasts.items():
        plt.plot(xs_te, v, label=k)
    plt.title(title); plt.xlabel("t"); plt.legend()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=120)
    plt.show(); plt.close()