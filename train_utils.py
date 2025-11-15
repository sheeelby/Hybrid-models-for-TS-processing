"""Backward-compatible shim to the new hybridts.training module."""
from __future__ import annotations

from project_paths import ensure_src_on_path

ensure_src_on_path()

from hybridts.training import TrainConfig, WindowDatasetStd, train_model

__all__ = ["WindowDatasetStd", "TrainConfig", "train_model"]
