"""Unit tests for quality gate evaluation."""

from __future__ import annotations

import pandas as pd

from house_price_ml.evaluation.gate_config import QualityGatesConfig
from house_price_ml.evaluation.gates import evaluate_training_gates


def _segment(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def test_gates_pass_when_metrics_good():
    summary = {
        "test_metrics": {"mae": 40000, "pct_within_10pct": 0.65},
        "baseline_metrics": {"mae": 50000},
        "beats_baseline": True,
        "walk_forward_model_mae_mean": 42000,
        "walk_forward_baseline_mae_mean": 50000,
    }
    region = _segment([{"segment": "Utrecht", "mae": 41000, "sample_size": 20}])
    prop = _segment([{"segment": "apartment", "mae": 39000, "sample_size": 25}])
    result = evaluate_training_gates(summary, region, prop)
    assert result.passed is True
    assert result.failures == []


def test_gates_fail_when_not_beating_baseline():
    summary = {
        "test_metrics": {"mae": 60000, "pct_within_10pct": 0.65},
        "baseline_metrics": {"mae": 50000},
        "beats_baseline": False,
        "walk_forward_model_mae_mean": 42000,
        "walk_forward_baseline_mae_mean": 50000,
    }
    region = _segment([{"segment": "Utrecht", "mae": 41000, "sample_size": 20}])
    prop = _segment([{"segment": "apartment", "mae": 39000, "sample_size": 25}])
    gates = QualityGatesConfig()
    result = evaluate_training_gates(summary, region, prop, gates=gates)
    assert result.passed is False
    assert any("baseline" in f for f in result.failures)


def test_gates_fail_segment_degradation():
    summary = {
        "test_metrics": {"mae": 40000, "pct_within_10pct": 0.65},
        "baseline_metrics": {"mae": 50000},
        "beats_baseline": True,
        "walk_forward_model_mae_mean": 42000,
        "walk_forward_baseline_mae_mean": 50000,
    }
    region = _segment([{"segment": "Amsterdam", "mae": 60000, "sample_size": 30}])
    prop = _segment([{"segment": "apartment", "mae": 39000, "sample_size": 25}])
    result = evaluate_training_gates(summary, region, prop)
    assert result.passed is False
    assert any("region" in f for f in result.failures)


def test_gates_fail_pct_within_tolerance():
    summary = {
        "test_metrics": {"mae": 40000, "pct_within_10pct": 0.30},
        "baseline_metrics": {"mae": 50000},
        "beats_baseline": True,
        "walk_forward_model_mae_mean": 42000,
        "walk_forward_baseline_mae_mean": 50000,
    }
    region = _segment([{"segment": "Utrecht", "mae": 41000, "sample_size": 20}])
    prop = _segment([{"segment": "apartment", "mae": 39000, "sample_size": 25}])
    gates = QualityGatesConfig()
    result = evaluate_training_gates(summary, region, prop, gates=gates)
    assert result.passed is False
    assert any("pct_within_10pct" in f for f in result.failures)
