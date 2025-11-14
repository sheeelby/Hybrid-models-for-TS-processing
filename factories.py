"""Shim to preserve the old factories import path."""
from __future__ import annotations

from project_paths import ensure_src_on_path

ensure_src_on_path()

from hybridts.models import make_model

__all__ = ["make_model"]
