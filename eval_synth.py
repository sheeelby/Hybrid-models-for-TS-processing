from __future__ import annotations

import argparse
import json
from inspect import signature
from pathlib import Path

from project_paths import ensure_src_on_path

ensure_src_on_path()

from hybridts.pipelines import evaluate_synth_hybrids  # noqa: E402

DEFAULT_CONFIG = Path("configs/synth_eval.json")


def load_config(path: Path | None):
    if path is None or not path.exists():
        return {}
    with open(path) as fh:
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

