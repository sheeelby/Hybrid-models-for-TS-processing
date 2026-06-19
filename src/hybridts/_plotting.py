"""Shared plotting helpers."""
from __future__ import annotations

from typing import Any


def _flatten_axes(axes: Any) -> list[Any]:
    if hasattr(axes, "ravel"):
        flattened = axes.ravel()
        return list(flattened.tolist() if hasattr(flattened, "tolist") else flattened)
    if isinstance(axes, (list, tuple)):
        return list(axes)
    return [axes]


def place_legend_below(
    fig: Any,
    axes: Any,
    *,
    ncol: int = 2,
    fontsize: float = 8,
    frameon: bool = False,
    top: float = 0.96,
    bottom_pad: float = 0.02,
):
    """Render a single deduplicated legend under the subplot area."""
    handles: list[Any] = []
    labels: list[str] = []
    seen_labels: set[str] = set()

    for ax in _flatten_axes(axes):
        ax_handles, ax_labels = ax.get_legend_handles_labels()
        for handle, label in zip(ax_handles, ax_labels):
            if not label or label.startswith("_") or label in seen_labels:
                continue
            handles.append(handle)
            labels.append(label)
            seen_labels.add(label)

    if not handles:
        fig.tight_layout(rect=(0.0, 0.0, 1.0, top))
        return None

    legend = fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.0),
        ncol=max(1, int(ncol)),
        fontsize=fontsize,
        frameon=frameon,
    )
    fig.canvas.draw()
    legend_bbox = legend.get_window_extent(renderer=fig.canvas.get_renderer())
    legend_bbox = legend_bbox.transformed(fig.transFigure.inverted())
    bottom = min(0.45, max(bottom_pad, legend_bbox.height + bottom_pad))
    fig.tight_layout(rect=(0.0, bottom, 1.0, top))
    return legend
