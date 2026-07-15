"""MLflow tracking configuration (Databricks-hosted by default)."""

from __future__ import annotations

import os

import mlflow

from house_price_ml.config.settings import Settings


def configure_mlflow(settings: Settings, experiment_name: str | None = None) -> str:
    """Point MLflow at Databricks when credentials exist; else local file store (tests)."""
    if settings.databricks_host and settings.databricks_token:
        os.environ.setdefault("DATABRICKS_HOST", settings.databricks_host.rstrip("/"))
        os.environ.setdefault("DATABRICKS_TOKEN", settings.databricks_token)
        tracking_uri = settings.mlflow_tracking_uri or "databricks"
        registry_uri = settings.mlflow_registry_uri or "databricks-uc"
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_registry_uri(registry_uri)
    else:
        tracking_uri = settings.mlflow_tracking_uri or "sqlite:///mlruns/mlflow.db"
        mlflow.set_tracking_uri(tracking_uri)

    resolved_experiment = experiment_name or settings.mlflow_experiment_name
    mlflow.set_experiment(resolved_experiment)
    tracking_uri = str(mlflow.get_tracking_uri())
    print(
        f"MLflow tracking: {tracking_uri} → experiment {resolved_experiment}",
        flush=True,
    )
    return tracking_uri
