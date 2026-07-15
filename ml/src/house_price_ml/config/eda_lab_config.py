"""Load EDA lab playbook settings from YAML."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field

from house_price_ml.config.training_config import ModelType

_CONFIG_PKG = Path(__file__).resolve().parents[1] / "config"
_ML_ROOT = Path(__file__).resolve().parents[3]
_REPO_DEFAULT = _ML_ROOT / "config" / "eda_lab.yaml"
_BUNDLED_DEFAULT = _CONFIG_PKG / "eda_lab.yaml"


class SectionToggles(BaseModel):
    data_quality: bool = True
    univariate: bool = True
    bivariate: bool = True
    correlation: bool = True
    business_hypotheses: bool = True
    feature_matrix: bool = True
    model_selection: bool = True
    residual_analysis: bool = True
    shap: bool = True


class DataQualitySection(BaseModel):
    max_null_rate: float = 0.10
    max_reject_rate: float = 0.05
    key_columns: list[str] = Field(
        default_factory=lambda: [
            "region",
            "property_type",
            "surface_area",
            "label_sale_price",
            "sale_date",
        ]
    )


class UnivariateSection(BaseModel):
    numeric_columns: list[str] = Field(
        default_factory=lambda: [
            "surface_area",
            "number_of_rooms",
            "number_of_bedrooms",
            "build_year",
            "label_sale_price",
        ]
    )
    categorical_columns: list[str] = Field(
        default_factory=lambda: ["region", "property_type", "energy_label"]
    )
    outlier_iqr_multiplier: float = 1.5
    skew_warning_threshold: float = 1.0


class BivariateSection(BaseModel):
    numeric_vs_target: list[str] = Field(
        default_factory=lambda: ["surface_area", "number_of_rooms", "build_year"]
    )
    categorical_vs_target: list[str] = Field(
        default_factory=lambda: ["region", "property_type", "energy_label"]
    )
    segment_columns: list[str] = Field(default_factory=lambda: ["region", "property_type"])


class CorrelationSection(BaseModel):
    method: Literal["pearson", "spearman"] = "pearson"
    numeric_columns: list[str] = Field(default_factory=list)
    high_correlation_threshold: float = 0.85
    min_abs_correlation_with_target: float = 0.05


class BusinessHypothesis(BaseModel):
    name: str
    type: Literal["correlation", "segment_mean_ratio"]
    x: str | None = None
    y: str | None = None
    min_correlation: float | None = None
    segment_column: str | None = None
    segment_value: str | None = None
    target: str | None = None
    min_ratio_vs_overall: float | None = None


class FeatureMatrixExperiment(BaseModel):
    name: str
    include_groups: list[str]


class FeatureMatrixSection(BaseModel):
    enabled: bool = True
    max_train_rows: int = 3000
    model_type: ModelType = "random_forest"
    experiments: list[FeatureMatrixExperiment] = Field(default_factory=list)


class ModelSelectionSection(BaseModel):
    enabled: bool = True
    candidates: list[ModelType] = Field(default_factory=lambda: ["ridge", "random_forest"])
    max_train_rows: int = 3000


class ResidualAnalysisSection(BaseModel):
    segment_columns: list[str] = Field(default_factory=lambda: ["region", "property_type"])
    top_n_worst: int = 15


class ShapSection(BaseModel):
    enabled: bool = True
    max_samples: int = 200
    max_features: int = 15


class EdaLabConfig(BaseModel):
    sections: SectionToggles = Field(default_factory=SectionToggles)
    data_quality: DataQualitySection = Field(default_factory=DataQualitySection)
    univariate: UnivariateSection = Field(default_factory=UnivariateSection)
    bivariate: BivariateSection = Field(default_factory=BivariateSection)
    correlation: CorrelationSection = Field(default_factory=CorrelationSection)
    business_hypotheses: list[BusinessHypothesis] = Field(default_factory=list)
    feature_matrix: FeatureMatrixSection = Field(default_factory=FeatureMatrixSection)
    model_selection: ModelSelectionSection = Field(default_factory=ModelSelectionSection)
    residual_analysis: ResidualAnalysisSection = Field(default_factory=ResidualAnalysisSection)
    shap: ShapSection = Field(default_factory=ShapSection)


def resolve_eda_lab_config_path(explicit: Path | str | None = None) -> Path:
    if explicit is not None:
        path = Path(explicit)
        if not path.is_file():
            raise FileNotFoundError(f"EDA lab config not found: {path}")
        return path

    env_path = os.environ.get("EDA_LAB_CONFIG_PATH", "").strip()
    if env_path:
        path = Path(env_path)
        if not path.is_file():
            raise FileNotFoundError(f"EDA lab config not found: {path}")
        return path

    if _REPO_DEFAULT.is_file():
        return _REPO_DEFAULT
    if _BUNDLED_DEFAULT.is_file():
        return _BUNDLED_DEFAULT

    raise FileNotFoundError("No EDA lab config found. Expected ml/config/eda_lab.yaml.")


def load_eda_lab_config(path: Path | str | None = None) -> EdaLabConfig:
    config_path = resolve_eda_lab_config_path(path)
    raw = yaml.safe_load(config_path.read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"EDA lab config must be a YAML mapping: {config_path}")
    return EdaLabConfig.model_validate(raw)


@lru_cache
def get_eda_lab_config(path: str | None = None) -> EdaLabConfig:
    return load_eda_lab_config(path)
