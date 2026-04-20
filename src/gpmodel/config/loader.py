"""YAML loader + Pydantic validation for AppConfig."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from gpmodel.config.schema import AppConfig


def load_config(path: str | Path) -> AppConfig:
    """Load and validate an AppConfig from a YAML file."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {p}")
    data: dict[str, Any] = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    return AppConfig.model_validate(data)
