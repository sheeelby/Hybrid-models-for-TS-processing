from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True)
class DecompositionSpec:
    method: str = "modwt"
    wavelet: str = "db4"
    level: int = 1
    boundary: str = "wrap"
    seasonal_period: int | None = None
    stl_kwargs: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        method = str(self.method or "modwt").strip().lower()
        if method not in {"modwt", "stl"}:
            raise ValueError(f"Unsupported decomposition method '{self.method}'")
        object.__setattr__(self, "method", method)
        object.__setattr__(self, "wavelet", str(self.wavelet or "db4"))
        object.__setattr__(self, "level", int(max(1, self.level)))
        object.__setattr__(self, "boundary", str(self.boundary or "wrap").strip().lower())
        if self.seasonal_period is None:
            object.__setattr__(self, "seasonal_period", None)
        else:
            object.__setattr__(self, "seasonal_period", int(self.seasonal_period))
        object.__setattr__(self, "stl_kwargs", dict(self.stl_kwargs or {}))

    @property
    def group_key(self) -> str:
        return self.method

    def cache_key(self) -> tuple[Any, ...]:
        stl_items = tuple(sorted((str(k), repr(v)) for k, v in dict(self.stl_kwargs or {}).items()))
        return (
            self.method,
            self.wavelet,
            int(self.level),
            self.boundary,
            self.seasonal_period,
            stl_items,
        )


@dataclass(frozen=True)
class DecompositionResult:
    spec: DecompositionSpec
    names: tuple[str, ...]
    components: tuple[np.ndarray, ...]


def _stl_component_names() -> tuple[str, str, str]:
    return ("trend", "seasonal", "resid")


def describe_decomposition(spec: DecompositionSpec) -> str:
    if spec.method == "modwt":
        return f"MODWT ({spec.wavelet}, level={spec.level}, boundary={spec.boundary})"

    extras: list[str] = []
    period = dict(spec.stl_kwargs or {}).get("period", spec.seasonal_period)
    if period is not None:
        extras.append(f"period={int(period)}")
    for key in ("seasonal", "trend", "low_pass", "robust", "seasonal_deg", "trend_deg", "low_pass_deg"):
        if key in dict(spec.stl_kwargs or {}):
            extras.append(f"{key}={dict(spec.stl_kwargs or {})[key]}")
    if not extras:
        return "STL"
    return f"STL ({', '.join(extras)})"


def _stl_decompose(y: np.ndarray, spec: DecompositionSpec) -> DecompositionResult:
    from statsmodels.tsa.seasonal import STL

    y = np.asarray(y, float).ravel()
    zeros = np.zeros_like(y, dtype=float)
    names = _stl_component_names()

    kwargs = dict(spec.stl_kwargs or {})
    period_raw = kwargs.pop("period", spec.seasonal_period)
    period = int(period_raw) if period_raw is not None else None

    if y.size == 0:
        return DecompositionResult(spec=spec, names=names, components=(y.copy(), zeros.copy(), zeros.copy()))

    if period is None or period <= 1 or y.size < max(8, 2 * period):
        trend = y.astype(float, copy=True)
        return DecompositionResult(
            spec=spec,
            names=names,
            components=(trend, zeros.copy(), zeros.copy()),
        )

    fit = STL(y, period=period, **kwargs).fit()
    trend = np.asarray(fit.trend, float)
    seasonal = np.asarray(fit.seasonal, float)
    resid = np.asarray(fit.resid, float)
    return DecompositionResult(
        spec=spec,
        names=names,
        components=(trend, seasonal, resid),
    )


def _modwt_decompose(y: np.ndarray, spec: DecompositionSpec, *, check: bool) -> DecompositionResult:
    from .modwt_hybrid import modwt_decompose_with_boundary

    A, D = modwt_decompose_with_boundary(
        y,
        wavelet=spec.wavelet,
        level=spec.level,
        boundary=spec.boundary,
        check=check,
    )
    components = [np.asarray(A, float)] + [np.asarray(comp, float) for comp in D]
    names = tuple(["A_J"] + [f"D_{j}" for j in range(1, len(components))])
    return DecompositionResult(
        spec=spec,
        names=names,
        components=tuple(components),
    )


def decompose_series(
    y,
    *,
    spec: DecompositionSpec,
    check: bool = True,
) -> DecompositionResult:
    arr = np.asarray(y, float).ravel()
    if spec.method == "stl":
        return _stl_decompose(arr, spec)
    return _modwt_decompose(arr, spec, check=check)


__all__ = [
    "DecompositionResult",
    "DecompositionSpec",
    "decompose_series",
    "describe_decomposition",
]
