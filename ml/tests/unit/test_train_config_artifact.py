"""MLflow artifact logging for resolved training config."""

from __future__ import annotations

from pathlib import Path

import mlflow
import yaml
from mlflow.tracking import MlflowClient

from house_price_ml.data.synthetic import generate_listings
from house_price_ml.data.training_data import build_training_export
from house_price_ml.models.train import train


def _latest_run_id() -> str:
    runs = MlflowClient().search_runs(
        experiment_ids=[mlflow.get_experiment_by_name("/Shared/house_price_prediction").experiment_id],
        order_by=["start_time DESC"],
        max_results=1,
    )
    assert runs, "expected at least one MLflow run"
    return runs[0].info.run_id


def test_train_logs_resolved_training_config_artifact(tmp_path: Path) -> None:
    bronze_path = tmp_path / "listings.csv"
    generate_listings(120, seed=11).to_csv(bronze_path, index=False)
    export_path = tmp_path / "training_frame.parquet"
    build_training_export(bronze_path, export_path)
    out = tmp_path / "model"

    train(export_path, "ridge", out)

    run_id = _latest_run_id()
    client = MlflowClient()
    artifacts = [a.path for a in client.list_artifacts(run_id, "reports")]
    assert "reports/training_config.yaml" in artifacts

    download_dir = tmp_path / "artifacts"
    local_path = client.download_artifacts(run_id, "reports/training_config.yaml", str(download_dir))
    resolved = yaml.safe_load(Path(local_path).read_text())
    assert resolved["model_type"] == "ridge"
    assert resolved["ridge"]["alpha"] == 1.0
