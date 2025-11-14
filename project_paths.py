"""Bootstrap helpers for adding the src/ directory to sys.path."""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"


def ensure_src_on_path() -> Path:
    if str(SRC_DIR) not in sys.path:
        sys.path.insert(0, str(SRC_DIR))
    return SRC_DIR


# Automatically expose the package when this module is imported.
ensure_src_on_path()

__all__ = ["SRC_DIR", "PROJECT_ROOT", "ensure_src_on_path"]
