"""Shim to the new hybridts.hybrids.modwt module."""
from __future__ import annotations

from project_paths import ensure_src_on_path

ensure_src_on_path()

from hybridts.hybrids.modwt import *  # noqa: F401,F403
