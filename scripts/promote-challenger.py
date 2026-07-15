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


def _run_passes_gates(client, run_id: str) -> tuple[bool, str]:
    run = client.get_run(run_id)
    metrics = run.data.metrics
    tags = run.data.tags

    beats = metrics.get("beats_baseline")
    if beats is None:
        beats_tag = tags.get("beats_baseline", "false").lower()
        beats = 1.0 if beats_tag in ("true", "1", "yes") else 0.0

    gates = metrics.get("gates_passed")
    if gates is None:
        gates_tag = tags.get("gates_passed", "false").lower()
        gates = 1.0 if gates_tag in ("true", "1", "yes") else 0.0

    if beats < 1.0:
        return False, "beats_baseline=0"
    if gates < 1.0:
        failures = tags.get("gate_failures", "unknown")
        return False, f"gates_passed=0 ({failures})"
    return True, ""


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Register an MLflow run and set @challenger (does not deploy serving)."
    )
    parser.add_argument("--run-id", required=True, help="MLflow run ID from Experiments UI")
    parser.add_argument("--catalog", default=None, help="Unity Catalog (default: DATABRICKS_CATALOG)")
    parser.add_argument("--alias", default="challenger", help="Registry alias to set")
    parser.add_argument(
        "--skip-gate-check",
        action="store_true",
        help="Allow promotion even when quality gates failed (not recommended).",
    )
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

    client = MlflowClient(registry_uri="databricks-uc")
    if not args.skip_gate_check:
        ok, reason = _run_passes_gates(client, args.run_id)
        if not ok:
            print(f"ERROR: Refusing promotion — run failed quality gates ({reason})", file=sys.stderr)
            return 1

    model_name = f"{catalog}.{schema}.house_price_model"
    model_uri = f"runs:/{args.run_id}/model"

    print(f"==> Register {model_uri} as {model_name}")
    registered = mlflow.register_model(model_uri=model_uri, name=model_name)
    version = str(registered.version)
    print(f"    Registered version {version}")

    client.set_registered_model_alias(model_name, alias, version)
    print(f"    Alias @{alias} -> version {version}")
    print("")
    print("Next: refresh the staging endpoint (includes inference verification)")
    print("  make deploy-serving-from-registry")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
