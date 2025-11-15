"""Hybrid time-series modeling utilities.

This package groups reusable components (config, data access,
models, hybrids, training utilities, and experiment pipelines)
so future neural models like Helformer can be plugged in easily.
"""

from .config.settings import settings

__all__ = ["settings"]
