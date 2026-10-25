"""Loads configs/base.yaml and turns relative paths into absolute ones."""
from __future__ import annotations

from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_config(path: str | Path | None = None) -> dict:
    path = Path(path) if path else PROJECT_ROOT / "configs" / "base.yaml"
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for key in ("raw_file", "processed_dir", "artifacts_dir", "reports_dir"):
        p = Path(cfg["data"][key])
        cfg["data"][key] = str(p if p.is_absolute() else PROJECT_ROOT / p)
    return cfg
