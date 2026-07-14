"""Model regression tests against golden dataset."""

from pathlib import Path

import json
import pytest

from house_price_ml.models.train import train

GOLDEN_DATA = Path(__file__).resolve().parents[3] / "data" / "sample" / "listings.csv"
CHAMPION_MAE_THRESHOLD = 1.10  # candidate MAE must be within 110% of baseline


@pytest.fixture(scope="module")
def training_summary(tmp_path_factory):
    if not GOLDEN_DATA.exists():
        pytest.skip("Golden dataset not found — run make seed")
    out = tmp_path_factory.mktemp("model")
    train(GOLDEN_DATA, "random_forest", out)
    return json.loads((out / "training_summary.json").read_text())


def test_beats_business_baseline(training_summary):
    assert training_summary["beats_baseline"] is True


def test_performance_within_threshold(training_summary):
    test_mae = training_summary["test_metrics"]["mae"]
    baseline_mae = training_summary["baseline_metrics"]["mae"]
    assert test_mae <= baseline_mae * CHAMPION_MAE_THRESHOLD
