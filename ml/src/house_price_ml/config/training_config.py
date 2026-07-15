"""Load training hyperparameters and split settings from YAML."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge

ModelType = Literal["ridge", "random_forest"]

_PACKAGE_DIR = Path(__file__).resolve().parent
_ML_ROOT = Path(__file__).resolve().parents[3]
_REPO_DEFAULT = _ML_ROOT / "config" / "training.yaml"
_BUNDLED_DEFAULT = _PACKAGE_DIR / "training.yaml"


class RandomForestParams(BaseModel):
    n_estimators: int = 200
    max_depth: int = 12
    random_state: int = 42
    n_jobs: int = -1


class RidgeParams(BaseModel):
    alpha: float = 1.0


class SplitParams(BaseModel):
    walk_forward_folds: int = Field(default=3, ge=1)
    walk_forward_test_quarters: int = Field(default=1, ge=1)
    holdout_test_quarters: int = Field(default=2, ge=1)


class TrainingConfig(BaseModel):
    model_type: ModelType = "random_forest"
    random_forest: RandomForestParams = Field(default_factory=RandomForestParams)
    ridge: RidgeParams = Field(default_factory=RidgeParams)
    splits: SplitParams = Field(default_factory=SplitParams)

    def make_estimator(self) -> BaseEstimator:
        if self.model_type == "ridge":
            return Ridge(**self.ridge.model_dump())
        return RandomForestRegressor(**self.random_forest.model_dump())

    def mlflow_params(self) -> dict[str, Any]:
        params: dict[str, Any] = {
            "model_type": self.model_type,
            "holdout_test_quarters": self.splits.holdout_test_quarters,
            "walk_forward_folds": self.splits.walk_forward_folds,
            "walk_forward_test_quarters": self.splits.walk_forward_test_quarters,
        }
        if self.model_type == "ridge":
            params.update({f"ridge_{k}": v for k, v in self.ridge.model_dump().items()})
        else:
            params.update({f"rf_{k}": v for k, v in self.random_forest.model_dump().items()})
        return params


def resolve_training_config_path(explicit: Path | str | None = None) -> Path:
    if explicit is not None:
        path = Path(explicit)
        if not path.is_file():
            raise FileNotFoundError(f"Training config not found: {path}")
        return path

    env_path = os.environ.get("TRAINING_CONFIG_PATH", "").strip()
    if env_path:
        path = Path(env_path)
        if not path.is_file():
            raise FileNotFoundError(f"Training config not found: {path}")
        return path

    if _REPO_DEFAULT.is_file():
        return _REPO_DEFAULT
    if _BUNDLED_DEFAULT.is_file():
        return _BUNDLED_DEFAULT

    raise FileNotFoundError(
        "No training config found. Expected ml/config/training.yaml or set TRAINING_CONFIG_PATH."
    )


def load_training_config(path: Path | str | None = None) -> TrainingConfig:
    config_path = resolve_training_config_path(path)
    raw = yaml.safe_load(config_path.read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"Training config must be a YAML mapping: {config_path}")
    return TrainingConfig.model_validate(raw)


@lru_cache
def get_training_config(path: str | None = None) -> TrainingConfig:
    return load_training_config(path)
