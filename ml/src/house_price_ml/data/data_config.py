"""Synthetic data profile loading."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import yaml

_ML_ROOT = Path(__file__).resolve().parents[3]
_DATA_CONFIG = _ML_ROOT / "config" / "data.yaml"


@dataclass
class DataProfile:
    name: str
    rows: int
    seed: int
    missing_rate: float = 0.0
    outlier_rate: float = 0.0
    invalid_rate: float = 0.01
    start_year: int = 2023
    span_days: int = 900


def load_data_profile(name: str | None = None) -> DataProfile:
    if not _DATA_CONFIG.is_file():
        return DataProfile(name="default", rows=500, seed=42)

    raw = yaml.safe_load(_DATA_CONFIG.read_text())
    default_name = os.environ.get("DATA_PROFILE", raw.get("default_profile", "engineering"))
    profile_name = name or default_name
    profiles = raw.get("profiles", {})
    if profile_name not in profiles:
        raise KeyError(f"Unknown data profile '{profile_name}'. Available: {list(profiles)}")
    cfg = profiles[profile_name]
    return DataProfile(name=profile_name, **cfg)
