"""Tests for Databricks train job helpers."""

from __future__ import annotations

from house_price_ml.config.training_config import load_training_config
from house_price_ml.jobs.databricks_train import (
    apply_experiment_flags,
    parse_bool,
    training_config_from_job_params,
)


def test_parse_bool():
    assert parse_bool("true") is True
    assert parse_bool("false") is False
    assert parse_bool("", default=True) is True


def test_apply_experiment_flags():
    base = load_training_config()
    assert base.tuning.enabled is False
    tuned = apply_experiment_flags(base, enable_tuning=True, enable_ablation=True)
    assert tuned.tuning.enabled is True
    assert tuned.ablation.enabled is True
    assert tuned.explainability.enabled == base.explainability.enabled


def test_training_config_from_job_params():
    cfg = training_config_from_job_params(
        enable_tuning="true",
        enable_ablation="false",
        enable_explainability="yes",
    )
    assert cfg.tuning.enabled is True
    assert cfg.ablation.enabled is False
    assert cfg.explainability.enabled is True
