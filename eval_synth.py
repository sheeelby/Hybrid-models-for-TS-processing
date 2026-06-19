from __future__ import annotations

import argparse
import json
from inspect import signature
from pathlib import Path

from project_paths import ensure_src_on_path

ensure_src_on_path()

from hybridts.pipelines import evaluate_synth_hybrids  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "synth_eval.json"


def resolve_config_path(path: Path | None) -> Path | None:
    if path is None:
        return None
    if path.exists():
        return path
    if not path.is_absolute():
        alt = PROJECT_ROOT / "configs" / path.name
        if alt.exists():
            return alt
    return path


def load_config(path: Path | None):
    resolved = resolve_config_path(path)
    if resolved is None:
        return {}
    if not resolved.exists():
        raise FileNotFoundError(f"Config file not found: {resolved}")
    with open(resolved) as fh:
        return json.load(fh)


def main():
    parser = argparse.ArgumentParser(description="Evaluate hybrid models on synthetic time series")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Path to a JSON file with pipeline parameters",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    sig = signature(evaluate_synth_hybrids)
    allowed = {k for k in sig.parameters}
    filtered = {k: v for k, v in cfg.items() if k in allowed}

    evaluate_synth_hybrids(**filtered)


if __name__ == "__main__":
    main()

