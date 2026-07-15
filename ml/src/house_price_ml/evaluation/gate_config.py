"""Load quality gate thresholds from YAML."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

_CONFIG_PKG = Path(__file__).resolve().parents[1] / "config"
_ML_ROOT = Path(__file__).resolve().parents[3]
_REPO_DEFAULT = _ML_ROOT / "config" / "quality_gates.yaml"
_BUNDLED_DEFAULT = _CONFIG_PKG / "quality_gates.yaml"


class HoldoutGates(BaseModel):
    beats_baseline: bool = True
    max_mae_vs_baseline_ratio: float = 1.10
    min_pct_within_10pct: float = 0.50


class WalkForwardGates(BaseModel):
    model_beats_baseline: bool = True


class SegmentGates(BaseModel):
    max_mae_degradation_vs_overall: float = 1.15
    min_segment_sample_size: int = 15


class PromotionGates(BaseModel):
    max_mae_vs_champion_ratio: float = 1.10


class QualityGatesConfig(BaseModel):
    holdout: HoldoutGates = Field(default_factory=HoldoutGates)
    walk_forward: WalkForwardGates = Field(default_factory=WalkForwardGates)
    segments: SegmentGates = Field(default_factory=SegmentGates)
    promotion: PromotionGates = Field(default_factory=PromotionGates)


def resolve_quality_gates_path(explicit: Path | str | None = None) -> Path:
    if explicit is not None:
        path = Path(explicit)
        if not path.is_file():
            raise FileNotFoundError(f"Quality gates config not found: {path}")
        return path

    env_path = os.environ.get("QUALITY_GATES_PATH", "").strip()
    if env_path:
        path = Path(env_path)
        if not path.is_file():
            raise FileNotFoundError(f"Quality gates config not found: {path}")
        return path

    if _REPO_DEFAULT.is_file():
        return _REPO_DEFAULT
    if _BUNDLED_DEFAULT.is_file():
        return _BUNDLED_DEFAULT

    raise FileNotFoundError(
        "No quality gates config found. Expected ml/config/quality_gates.yaml."
    )


def load_quality_gates(path: Path | str | None = None) -> QualityGatesConfig:
    config_path = resolve_quality_gates_path(path)
    raw = yaml.safe_load(config_path.read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"Quality gates must be a YAML mapping: {config_path}")
    return QualityGatesConfig.model_validate(raw)


@lru_cache
def get_quality_gates(path: str | None = None) -> QualityGatesConfig:
    return load_quality_gates(path)
