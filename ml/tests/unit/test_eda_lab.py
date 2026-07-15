"""Unit tests for EDA lab helpers."""

from __future__ import annotations

import pandas as pd

from house_price_ml.config.eda_lab_config import load_eda_lab_config
from house_price_ml.evaluation.splits import holdout_test_split
from house_price_ml.jobs.eda_lab import (
    bivariate_vs_target,
    correlation_report,
    evaluate_business_hypotheses,
    evaluate_data_quality_gates,
    run_eda_playbook,
    run_feature_matrix,
    run_model_selection,
    univariate_profile,
)
from house_price_ml.jobs.experiment_lab import data_quality_summary, load_training_frame_sample


def _training_frame(n: int = 120) -> pd.DataFrame:
    return load_training_frame_sample(profile="demo", rows=n)


def test_load_eda_lab_config():
    cfg = load_eda_lab_config()
    assert cfg.sections.univariate is True
    assert len(cfg.business_hypotheses) >= 2


def test_univariate_profile_has_numeric_and_categorical():
    df = _training_frame(80)
    cfg = load_eda_lab_config()
    profile = univariate_profile(df, cfg)
    dtypes = set(profile["dtype"])
    assert "numeric" in dtypes
    assert "categorical" in dtypes


def test_bivariate_vs_target_correlation():
    df = _training_frame(100)
    cfg = load_eda_lab_config()
    table = bivariate_vs_target(df, cfg)
    surface = table[(table["feature"] == "surface_area") & (table["type"] == "numeric")]
    assert not surface.empty
    assert surface.iloc[0]["correlation_with_target"] is not None


def test_correlation_report_flags():
    df = _training_frame(100)
    cfg = load_eda_lab_config()
    matrix, flags = correlation_report(df, cfg)
    assert not matrix.empty
    assert "label_sale_price" in matrix.columns


def test_business_hypotheses_surface_price():
    df = _training_frame(100)
    cfg = load_eda_lab_config()
    results = evaluate_business_hypotheses(df, cfg)
    assert "price_increases_with_surface" in results["hypothesis"].values


def test_data_quality_gates_from_summary():
    df = _training_frame(50)
    summary = data_quality_summary(df, None)
    cfg = load_eda_lab_config()
    gates = evaluate_data_quality_gates(summary, cfg)
    assert gates["passed"].all()


def test_run_eda_playbook_keys():
    df = _training_frame(80)
    results = run_eda_playbook(df, dq_summary=data_quality_summary(df, None))
    assert "univariate" in results
    assert "correlation_matrix" in results


def test_feature_matrix_experiments(tmp_path):
    df = _training_frame(150)
    cfg = load_eda_lab_config()
    train_df, test_df = holdout_test_split(df, test_quarters=1)
    matrix = run_feature_matrix(train_df, test_df, cfg)
    assert len(matrix) == len(cfg.feature_matrix.experiments)
    assert "delta_mae_vs_full" in matrix.columns


def test_model_selection_orders_by_mae():
    df = _training_frame(150)
    cfg = load_eda_lab_config()
    train_df, test_df = holdout_test_split(df, test_quarters=1)
    comparison = run_model_selection(train_df, test_df, cfg)
    assert "business_baseline" in comparison["model"].values
    assert comparison.iloc[0]["mae"] <= comparison.iloc[-1]["mae"]
