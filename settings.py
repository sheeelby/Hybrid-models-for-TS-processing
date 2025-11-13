# settings.py
from pathlib import Path

BASE_DIR   = Path(r"C:\Users\Admin\Desktop\Hybrid-models-for-TS-processing").resolve()

# ГДЕ CSV (готовые TRAIN/TSTS)
M3_CSV_DIR = BASE_DIR / r"M3\none"

# ГДЕ TSF (сырьё для сборки CSV)
M3_TSF_DIR = BASE_DIR / "M3"

# Куда складывать графики/метрики
OUT_DIR = BASE_DIR / "out"
OUT_DIR.mkdir(parents=True, exist_ok=True)
