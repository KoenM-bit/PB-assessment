"""Integration tests for prediction flow."""

import json
from pathlib import Path

import pandas as pd
import pytest
from house_price_ml.data.synthetic import generate_listings
from house_price_ml.data.validation import validate_prediction_request
from house_price_ml.features.pipeline import compute_row_features, raw_to_feature_frame
from house_price_ml.models.train import train


@pytest.fixture
def sample_data(tmp_path):
    golden = Path(__file__).resolve().parents[3] / "data" / "sample" / "listings.csv"
    if golden.exists():
        return golden
    path = tmp_path / "listings.csv"
    generate_listings(500, seed=42).to_csv(path, index=False)
    return path


def test_validation_to_features_to_training(sample_data):
    df = pd.read_csv(sample_data)
    row = df.iloc[0].to_dict()
    validation = validate_prediction_request(
        {
            "surface_area": row["surface_area"],
            "number_of_rooms": int(row["number_of_rooms"]),
            "number_of_bedrooms": int(row["number_of_bedrooms"]),
            "build_year": int(row["build_year"]),
            "energy_label": row["energy_label"],
            "property_type": row["property_type"],
            "garden": bool(row["garden"]),
            "region": row["region"],
            "latitude": row["latitude"],
            "longitude": row["longitude"],
        }
    )
    assert validation.is_valid
    features = compute_row_features({**row, "prediction_date": "2026-01-01"})
    assert features["house_age"] > 0
    assert features["surface_per_room"] > 0


def test_full_training_pipeline(sample_data, tmp_path):
    out = tmp_path / "model"
    result = train(sample_data, "random_forest", out)
    summary = json.loads((result / "training_summary.json").read_text())
    assert "test_metrics" in summary
    assert "baseline_metrics" in summary
    assert (result / "mlflow_model").exists()


def test_feature_frame_batch(sample_data):
    df = pd.read_csv(sample_data).head(10)
    frame = raw_to_feature_frame(df.to_dict("records"))
    assert len(frame) == 10
    assert "region_median_price_per_sqm" in frame.columns
