"""Unit tests for experiment lab helpers."""

from __future__ import annotations

import pandas as pd

from house_price_ml.evaluation.gate_config import QualityGatesConfig
from house_price_ml.evaluation.gates import GateResult, evaluate_training_gates
from house_price_ml.jobs.databricks_train import parse_bool
from house_price_ml.jobs.experiment_lab import (
    baseline_holdout_metrics,
    data_quality_summary,
    format_gate_report,
    load_training_frame_sample,
    segment_balance_report,
)


def test_data_quality_summary_with_rejects():
    silver = pd.DataFrame(
        {
            "region": ["Utrecht", "Amsterdam", None],
            "property_type": ["apartment", "terraced_house", "apartment"],
            "surface_area": [80, 100, 90],
            "label_sale_price": [300000, 400000, 350000],
            "sale_date": pd.to_datetime(["2023-01-01", "2023-02-01", "2023-03-01"]),
        }
    )
    rejected = pd.DataFrame({"listing_id": ["x1", "x2"]})
    summary = data_quality_summary(silver, rejected)
    assert summary["silver_rows"] == 3
    assert summary["rejected_rows"] == 2
    assert summary["bronze_rows"] == 5
    assert summary["null_rates"]["region"] == round(1 / 3, 4)


def test_segment_balance_report_includes_price_category():
    df = pd.DataFrame(
        {
            "region": ["Utrecht", "Utrecht", "Amsterdam"],
            "property_type": ["apartment", "apartment", "terraced_house"],
            "label_sale_price": [250000, 260000, 800000],
        }
    )
    report = segment_balance_report(df)
    assert set(report["segment_type"]) == {"region", "property_type", "price_category"}
    for segment_type in ("region", "property_type", "price_category"):
        subset = report[report["segment_type"] == segment_type]
        assert subset["count"].sum() == len(df)


def test_format_gate_report_flattens_violations():
    summary = {
        "test_metrics": {"mae": 40000, "pct_within_10pct": 0.65},
        "baseline_metrics": {"mae": 50000},
        "beats_baseline": True,
        "walk_forward_model_mae_mean": 42000,
        "walk_forward_baseline_mae_mean": 50000,
    }
    region = pd.DataFrame([{"segment": "Amsterdam", "mae": 60000, "sample_size": 30}])
    prop = pd.DataFrame([{"segment": "apartment", "mae": 39000, "sample_size": 25}])
    gate_result = evaluate_training_gates(summary, region, prop)
    table = format_gate_report(gate_result)
    assert not table.empty
    assert any("Amsterdam" in str(v) for v in table["segment"].tolist())


def test_format_gate_report_passed():
    table = format_gate_report(GateResult(passed=True))
    assert table.iloc[0]["check"] == "all gates passed"


def test_baseline_holdout_metrics():
    train_df = load_training_frame_sample(profile="demo", rows=80)
    split_idx = int(len(train_df) * 0.8)
    train_slice = train_df.iloc[:split_idx]
    test_slice = train_df.iloc[split_idx:]
    metrics = baseline_holdout_metrics(train_slice, test_slice)
    assert metrics["baseline_mae"] > 0


def test_load_training_frame_sample_demo_profile():
    frame = load_training_frame_sample(profile="demo", rows=60)
    assert len(frame) > 0
    assert "label_sale_price" in frame.columns


def test_parse_bool_integration_for_lab_flags():
    assert parse_bool("false", default=True) is False
    assert parse_bool("true", default=False) is True
    gates = QualityGatesConfig()
    assert gates.segments.min_segment_sample_size == 15
