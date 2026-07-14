"""Model smoke and regression tests."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from house_price_ml.data.synthetic import generate_listings
from house_price_ml.features.aggregates import silver_to_gold_features
from house_price_ml.data.silver import bronze_to_silver
from house_price_ml.models.baseline import BusinessBaseline
from house_price_ml.models.train import train
from house_price_ml.serving.mlflow_model import build_sklearn_pipeline, save_model_artifact
from house_price_ml.serving.pyfunc_model import HousePriceModel  # noqa: F401 — tests
import mlflow.pyfunc


@pytest.fixture(scope="module")
def trained_model(tmp_path_factory):
    data_path = tmp_path_factory.mktemp("data") / "listings.csv"
    df = generate_listings(500, seed=42)
    df.to_csv(data_path, index=False)
    out = tmp_path_factory.mktemp("model")
    train(data_path, "random_forest", out)
    return out / "mlflow_model"


def test_baseline_predicts_positive(trained_model):
    df = generate_listings(20)
    baseline = BusinessBaseline().fit(df)
    preds = baseline.predict(df)
    assert all(p > 0 for p in preds)
    assert all(np.isfinite(p) for p in preds)


def test_model_load_and_predict(trained_model):
    model = mlflow.pyfunc.load_model(str(trained_model))
    meta = model.metadata
    assert meta.signature is not None
    assert meta.signature.inputs is not None
    assert meta.signature.outputs is not None
    sample = {
        "surface_area": 120.0,
        "number_of_rooms": 5,
        "number_of_bedrooms": 3,
        "build_year": 1985,
        "energy_label": "B",
        "property_type": "terraced_house",
        "garden": True,
        "region": "Utrecht",
        "latitude": 52.0907,
        "longitude": 5.1214,
        "prediction_date": "2026-07-14",
    }
    result = model.predict(pd.DataFrame([sample]))
    assert "predicted_price" in result.columns
    price = float(result["predicted_price"].iloc[0])
    assert price > 0
    assert np.isfinite(price)


def test_model_training_summary_exists(trained_model):
    import json
    summary_path = trained_model.parent / "training_summary.json"
    summary = json.loads(summary_path.read_text())
    assert "test_metrics" in summary
    assert "baseline_metrics" in summary


def test_robustness_small_house(trained_model):
    model = mlflow.pyfunc.load_model(str(trained_model))
    sample = {
        "surface_area": 45,
        "number_of_rooms": 2,
        "number_of_bedrooms": 1,
        "build_year": 2010,
        "energy_label": "A",
        "property_type": "apartment",
        "garden": False,
        "region": "Groningen",
        "latitude": 53.21,
        "longitude": 6.56,
        "prediction_date": "2026-07-14",
    }
    result = model.predict(pd.DataFrame([sample]))
    assert float(result["predicted_price"].iloc[0]) > 0


def test_robustness_large_house(trained_model):
    model = mlflow.pyfunc.load_model(str(trained_model))
    sample = {
        "surface_area": 250,
        "number_of_rooms": 8,
        "number_of_bedrooms": 5,
        "build_year": 1970,
        "energy_label": "D",
        "property_type": "detached",
        "garden": True,
        "region": "Amsterdam",
        "latitude": 52.37,
        "longitude": 4.90,
        "prediction_date": "2026-07-14",
    }
    result = model.predict(pd.DataFrame([sample]))
    assert float(result["predicted_price"].iloc[0]) > 0
