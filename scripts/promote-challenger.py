#!/usr/bin/env python3
"""Promote an MLflow experiment run to @challenger in Unity Catalog (staging go-live step)."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def load_env(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Register an MLflow run and set @challenger (does not deploy serving)."
    )
    parser.add_argument("--run-id", required=True, help="MLflow run ID from Experiments UI")
    parser.add_argument("--catalog", default=None, help="Unity Catalog (default: DATABRICKS_CATALOG)")
    parser.add_argument("--alias", default="challenger", help="Registry alias to set")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    load_env(root / ".env")

    host = os.environ.get("DATABRICKS_HOST", "").rstrip("/")
    token = os.environ.get("DATABRICKS_TOKEN", "")
    catalog = args.catalog or os.environ.get("DATABRICKS_CATALOG", "house_price_staging")
    schema = os.environ.get("DATABRICKS_SCHEMA", "gold")
    alias = args.alias

    if not host or not token:
        print("ERROR: Set DATABRICKS_HOST and DATABRICKS_TOKEN in .env", file=sys.stderr)
        return 1

    os.environ["DATABRICKS_HOST"] = host
    os.environ["DATABRICKS_TOKEN"] = token
    os.environ["MLFLOW_TRACKING_URI"] = "databricks"
    os.environ["MLFLOW_REGISTRY_URI"] = "databricks-uc"

    import mlflow
    from mlflow import MlflowClient

    mlflow.set_tracking_uri("databricks")
    mlflow.set_registry_uri("databricks-uc")

    model_name = f"{catalog}.{schema}.house_price_model"
    model_uri = f"runs:/{args.run_id}/model"

    print(f"==> Register {model_uri} as {model_name}")
    registered = mlflow.register_model(model_uri=model_uri, name=model_name)
    version = str(registered.version)
    print(f"    Registered version {version}")

    client = MlflowClient(registry_uri="databricks-uc")
    client.set_registered_model_alias(model_name, alias, version)
    print(f"    Alias @{alias} -> version {version}")
    print("")
    print("Next: refresh the staging endpoint (includes inference verification)")
    print("  make deploy-serving-from-registry")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
