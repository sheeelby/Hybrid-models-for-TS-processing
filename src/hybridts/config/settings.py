"""Environment-driven settings for the project."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_ENV_PATH = _PROJECT_ROOT / ".env"


def _read_env_file(path: Path) -> Dict[str, str]:
    if not path.exists():
        return {}
    data: Dict[str, str] = {}
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        data[key.strip()] = value.strip().strip('"').strip("'")
    return data


@dataclass
class Settings:
    project_root: Path
    data_dir: Path
    m3_csv_dir: Path
    m3_tsf_dir: Path
    outputs_dir: Path

    @classmethod
    def from_env(cls) -> "Settings":
        env = {**_read_env_file(_ENV_PATH), **os.environ}
        root = Path(env.get("PROJECT_ROOT", _PROJECT_ROOT)).resolve()
        data_dir = Path(env.get("DATA_DIR", root / "data")).expanduser().resolve()
        if not data_dir.exists():
            alt = (root / "src" / "data").resolve()
            if alt.exists():
                data_dir = alt
            else:
                data_dir.mkdir(parents=True, exist_ok=True)
        m3_csv_dir = Path(env.get("M3_CSV_DIR", data_dir / "m3" / "csv")).expanduser().resolve()
        m3_tsf_dir = Path(env.get("M3_TSF_DIR", data_dir / "m3" / "tsf")).expanduser().resolve()
        outputs_dir = Path(env.get("OUTPUTS_DIR", root / "outputs")).resolve()
        outputs_dir.mkdir(parents=True, exist_ok=True)
        return cls(
            project_root=root,
            data_dir=data_dir,
            m3_csv_dir=m3_csv_dir,
            m3_tsf_dir=m3_tsf_dir,
            outputs_dir=outputs_dir,
        )


settings = Settings.from_env()

__all__ = ["settings", "Settings"]
