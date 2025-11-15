"""Shim to the reorganized HybridPlus implementation."""
from __future__ import annotations

from project_paths import ensure_src_on_path

ensure_src_on_path()

from hybridts.hybrids.modwt_hybrid import HybridPlus, modwt_decompose

__all__ = ["HybridPlus", "modwt_decompose"]
