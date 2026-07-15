"""Training frame assembly and loading tests."""

from pathlib import Path

import pandas as pd
import pytest

from house_price_ml.data.silver import bronze_to_silver
from house_price_ml.data.synthetic import generate_listings
from house_price_ml.data.training_data import (
    TrainingFrameError,
    assemble_training_frame,
    build_training_export,
    is_raw_bronze_input,
    load_training_frame,
    validate_training_frame,
)
from house_price_ml.features.aggregates import silver_to_gold_features
from house_price_ml.models.train import train


def _build_frame(n: int = 100) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = generate_listings(n, seed=7)
    clean, _ = bronze_to_silver(raw)
    gold = silver_to_gold_features(clean)
    return clean, gold


def test_assemble_training_frame_joins_silver_and_gold():
    silver, gold = _build_frame()
    frame = assemble_training_frame(silver, gold)
    assert len(frame) == len(silver)
    assert "label_sale_price" in frame.columns
    assert "feature_snapshot_date" in frame.columns
    assert "surface_area" in frame.columns


def test_validate_training_frame_rejects_raw_bronze():
    raw = generate_listings(20)
    assert is_raw_bronze_input(raw) is True
    with pytest.raises(TrainingFrameError, match="make gold-export"):
        validate_training_frame(raw)


def test_load_training_frame_from_dataframe():
    silver, gold = _build_frame(50)
    frame = assemble_training_frame(silver, gold)
    loaded = load_training_frame(frame)
    assert len(loaded) == len(frame)


def test_build_training_export_writes_parquet_and_metadata(tmp_path: Path):
    bronze_path = tmp_path / "listings.csv"
    generate_listings(80, seed=3).to_csv(bronze_path, index=False)
    output_path = tmp_path / "training_frame.parquet"

    frame, metadata = build_training_export(bronze_path, output_path)

    assert output_path.is_file()
    assert metadata["rejected_rows"] >= 0
    assert metadata["training_rows"] == len(frame)
    loaded = load_training_frame(output_path)
    assert len(loaded) == len(frame)


def test_train_rejects_raw_listings_csv(tmp_path: Path):
    bronze_path = tmp_path / "listings.csv"
    generate_listings(60, seed=1).to_csv(bronze_path, index=False)
    out = tmp_path / "model"

    with pytest.raises(TrainingFrameError, match="make gold-export"):
        train(bronze_path, "random_forest", out)


def test_train_accepts_assembled_export(tmp_path: Path):
    bronze_path = tmp_path / "listings.csv"
    generate_listings(120, seed=5).to_csv(bronze_path, index=False)
    export_path = tmp_path / "training_frame.parquet"
    build_training_export(bronze_path, export_path)
    out = tmp_path / "model"

    result = train(export_path, "random_forest", out)
    assert (result / "training_summary.json").is_file()
