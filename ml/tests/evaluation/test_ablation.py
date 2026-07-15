"""Smoke test for feature ablation."""

from __future__ import annotations

import pytest

from house_price_ml.config.training_config import TrainingConfig
from house_price_ml.data.synthetic import generate_listings
from house_price_ml.data.training_data import build_training_export
from house_price_ml.evaluation.ablation import run_ablation
from house_price_ml.evaluation.splits import holdout_test_split


@pytest.fixture(scope="module")
def tiny_frames(tmp_path_factory):
    data_dir = tmp_path_factory.mktemp("data")
    bronze = data_dir / "listings.csv"
    generate_listings(120, seed=7).to_csv(bronze, index=False)
    export = data_dir / "training.parquet"
    build_training_export(bronze, export)
    from house_price_ml.data.training_data import load_training_frame

    frame = load_training_frame(export)
    train_df, test_df = holdout_test_split(frame, test_quarters=1)
    return train_df, test_df


def test_ablation_runs_on_tiny_frame(tiny_frames):
    train_df, test_df = tiny_frames
    config = TrainingConfig()
    report = run_ablation(train_df, test_df, config)
    assert "full" in report["feature_group"].values
    assert len(report) == 6  # full + 5 groups
