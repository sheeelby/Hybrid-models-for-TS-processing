"""Project settings shim to keep legacy imports alive."""
from __future__ import annotations

from project_paths import ensure_src_on_path

ensure_src_on_path()

from hybridts.config import settings as _settings

BASE_DIR = _settings.project_root
M3_CSV_DIR = _settings.m3_csv_dir
M3_TSF_DIR = _settings.m3_tsf_dir
OUT_DIR = _settings.outputs_dir

__all__ = ["BASE_DIR", "M3_CSV_DIR", "M3_TSF_DIR", "OUT_DIR"]
