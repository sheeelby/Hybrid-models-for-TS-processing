"""Small helpers for writing incremental CSV checkpoints during long runs."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd


def reset_csv(path: Path) -> None:
    if path.exists():
        path.unlink()


def append_row(path: Path, row: Mapping[str, Any], columns: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(columns))
        if write_header:
            writer.writeheader()
        out: dict[str, Any] = {}
        for col in columns:
            val = row.get(col)
            if val is None or pd.isna(val):
                out[col] = ""
            else:
                out[col] = val
        writer.writerow(out)

