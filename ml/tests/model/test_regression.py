"""Model regression tests against golden dataset."""

import json
from pathlib import Path

import pytest

from house_price_ml.models.train import train

GOLDEN_EXPORT = Path(__file__).resolve().parents[3] / "data" / "sample" / "training_frame.parquet"
GOLDEN_BRONZE = Path(__file__).resolve().parents[3] / "data" / "sample" / "listings.csv"
CI_GATES = Path(__file__).resolve().parents[3] / "ml" / "config" / "quality_gates_ci.yaml"
CHAMPION_MAE_THRESHOLD = 1.10  # candidate MAE must be within 110% of baseline


@pytest.fixture(scope="module")
def training_summary(tmp_path_factory):
    if not GOLDEN_EXPORT.is_file():
        if not GOLDEN_BRONZE.is_file():
            pytest.skip("Golden dataset not found — run make seed && make gold-export")
        from house_price_ml.data.training_data import build_training_export

        build_training_export(GOLDEN_BRONZE, GOLDEN_EXPORT)
    out = tmp_path_factory.mktemp("model")
    train(GOLDEN_EXPORT, "random_forest", out, gates_path=CI_GATES)
    return json.loads((out / "training_summary.json").read_text())


def test_beats_business_baseline(training_summary):
    assert training_summary["beats_baseline"] is True


def test_gates_passed(training_summary):
    assert training_summary.get("gates_passed") is True


def test_performance_within_threshold(training_summary):
    test_mae = training_summary["test_metrics"]["mae"]
    baseline_mae = training_summary["baseline_metrics"]["mae"]
    assert test_mae <= baseline_mae * CHAMPION_MAE_THRESHOLD
