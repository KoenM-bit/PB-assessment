#!/usr/bin/env python3
"""Promote @challenger to @champion and keep @previous_champion for rollback."""
from __future__ import annotations

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
    root = Path(__file__).resolve().parents[1]
    load_env(root / ".env")

    host = os.environ.get("DATABRICKS_HOST", "").rstrip("/")
    token = os.environ.get("DATABRICKS_TOKEN", "")
    catalog = os.environ.get("DATABRICKS_CATALOG", "house_price_staging")
    schema = os.environ.get("DATABRICKS_SCHEMA", "gold")
    source_alias = os.environ.get("PROMOTE_FROM_ALIAS", "challenger")
    target_alias = os.environ.get("PROMOTE_TO_ALIAS", "champion")
    rollback_alias = os.environ.get("PROMOTE_ROLLBACK_ALIAS", "previous_champion")

    if not host or not token:
        print("ERROR: Set DATABRICKS_HOST and DATABRICKS_TOKEN in .env", file=sys.stderr)
        return 1

    os.environ["DATABRICKS_HOST"] = host
    os.environ["DATABRICKS_TOKEN"] = token
    os.environ["MLFLOW_REGISTRY_URI"] = "databricks-uc"

    from mlflow import MlflowClient
    from mlflow.exceptions import RestException

    model_name = f"{catalog}.{schema}.house_price_model"
    client = MlflowClient(registry_uri="databricks-uc")

    try:
        candidate = client.get_model_version_by_alias(model_name, source_alias)
    except RestException as exc:
        print(f"ERROR: No '{source_alias}' alias on {model_name}: {exc}", file=sys.stderr)
        return 1

    version = str(candidate.version)
    print(f"==> Promote {model_name}")
    print(f"    {source_alias} -> version {version}")

    try:
        current = client.get_model_version_by_alias(model_name, target_alias)
        client.set_registered_model_alias(model_name, rollback_alias, str(current.version))
        print(f"    {rollback_alias} <- former {target_alias} (version {current.version})")
    except RestException:
        print(f"    No existing {target_alias} alias — skip {rollback_alias}")

    client.set_registered_model_alias(model_name, target_alias, version)
    print(f"    {target_alias} <- version {version}")
    print("")
    print("Done. Update the serving endpoint to this version if needed:")
    print(f"  DATABRICKS_CATALOG={catalog} MODEL_ALIAS={target_alias} make deploy-serving")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
