"""MLflow logging of Delta table versions during training."""

from __future__ import annotations

import json
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient

from house_price_ml.data.synthetic import generate_listings
from house_price_ml.data.training_data import build_training_export
from house_price_ml.models.train import train


def test_train_logs_table_versions_to_mlflow(tmp_path: Path) -> None:
    bronze_path = tmp_path / "listings.csv"
    generate_listings(120, seed=9).to_csv(bronze_path, index=False)
    export_path = tmp_path / "training_frame.parquet"
    build_training_export(bronze_path, export_path)
    out = tmp_path / "model"

    table_versions = {
        "catalog.silver.listings_clean": 3,
        "catalog.gold.listing_features": 7,
    }
    train(
        export_path,
        "random_forest",
        out,
        table_versions=table_versions,
    )

    runs = MlflowClient().search_runs(
        experiment_ids=[mlflow.get_experiment_by_name("/Shared/house_price_prediction").experiment_id],
        order_by=["start_time DESC"],
        max_results=1,
    )
    assert runs, "expected at least one MLflow run"
    params = runs[0].data.params

    assert params["training_table_versions"] == json.dumps(table_versions, sort_keys=True)
    assert params["silver_table_version"] == "3"
    assert params["gold_table_version"] == "7"
