from __future__ import annotations

from dataclasses import dataclass


BASE_MODELS = {"timesnet", "nbeats"}

LEGACY_LABELS = {
    "timesnet": "TimesNet+",
    "nbeats": "N-BEATS Full",
    "raw_timesnet": "TimesNet (raw)",
    "raw_nbeats": "N-BEATS (raw)",
    "vw_timesnet_ets": "VW + TimesNet + ETS",
    "vw_timesnet_arima_auto": "VW + TimesNet + Arima_auto",
    "vw_nbeats_ets": "VW + N-Beats + ETS",
    "vw_nbeats_arima_auto": "VW + N-Beats + Arima_auto",
}


@dataclass(frozen=True)
class ModelSpec:
    name: str
    kind: str
    base_model_name: str
    label: str
    decomposition_method: str | None = None
    detail_method: str | None = None


def _base_label(name: str) -> str:
    return "TimesNet" if name == "timesnet" else "N-BEATS"


def normalize_model_name(name: str) -> str:
    value = str(name).strip().lower()
    if value.startswith("raw_"):
        return value
    if value.endswith("_raw"):
        base = value[: -len("_raw")]
        return f"raw_{base}"
    if value.startswith("direct_"):
        base = value[len("direct_") :]
        return f"raw_{base}"
    return value


def parse_model_spec(name: str) -> ModelSpec:
    value = normalize_model_name(name)
    if value.startswith("raw_"):
        base = value[len("raw_") :]
        if base in BASE_MODELS:
            return ModelSpec(
                name=value,
                kind="raw",
                base_model_name=base,
                label=LEGACY_LABELS.get(value, f"{_base_label(base)} (raw)"),
            )

    if value in BASE_MODELS:
        return ModelSpec(
            name=value,
            kind="hybrid_all",
            base_model_name=value,
            label=LEGACY_LABELS.get(value, f"{_base_label(value)}+"),
            decomposition_method="modwt",
        )

    if value.startswith("vw_"):
        rest = value[len("vw_") :]
        for suffix in ("_arima_auto", "_ets"):
            if rest.endswith(suffix):
                base = rest[: -len(suffix)]
                if base in BASE_MODELS:
                    detail = suffix[1:]
                    return ModelSpec(
                        name=value,
                        kind="hybrid_mixed",
                        base_model_name=base,
                        label=LEGACY_LABELS.get(
                            value,
                            f"VW + {_base_label(base)} + {detail.title()}",
                        ),
                        decomposition_method="modwt",
                        detail_method=detail,
                    )

    if value.startswith("stl_"):
        rest = value[len("stl_") :]
        if rest in BASE_MODELS:
            return ModelSpec(
                name=value,
                kind="hybrid_all",
                base_model_name=rest,
                label=f"STL + {_base_label(rest)}",
                decomposition_method="stl",
            )
        for suffix in ("_arima_auto", "_ets"):
            if rest.endswith(suffix):
                base = rest[: -len(suffix)]
                if base in BASE_MODELS:
                    detail = suffix[1:]
                    detail_label = "Arima_auto" if detail == "arima_auto" else "ETS"
                    return ModelSpec(
                        name=value,
                        kind="hybrid_mixed",
                        base_model_name=base,
                        label=f"STL + {_base_label(base)} + {detail_label}",
                        decomposition_method="stl",
                        detail_method=detail,
                    )

    raise ValueError(f"Unknown base model '{name}'")


__all__ = ["ModelSpec", "normalize_model_name", "parse_model_spec"]
