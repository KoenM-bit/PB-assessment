#!/usr/bin/env python3
"""Promote staging @challenger to production @champion with full traceability.

Copies the model version from the staging Unity Catalog registry into the
production registry, updates aliases (champion / previous_champion), and
deploys house-price-serving-prod.

Usage:
  CONFIRM_PROMOTE=yes make promote-to-production
  python scripts/promote-to-production.py --dry-run
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
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


def load_deploy_serving_module():
    path = Path(__file__).resolve().parent / "deploy-serving.py"
    spec = importlib.util.spec_from_file_location("deploy_serving", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def version_tags(client, model_name: str, version: str) -> dict[str, str]:
    try:
        mv = client.get_model_version(model_name, version)
        return dict(mv.tags or {})
    except Exception:
        return {}


def version_run_id(client, model_name: str, version: str) -> str | None:
    try:
        mv = client.get_model_version(model_name, version)
        return mv.run_id
    except Exception:
        return None


def run_test_mae(client, run_id: str | None) -> float | None:
    if not run_id:
        return None
    try:
        run = client.get_run(run_id)
        return run.data.metrics.get("test_mae")
    except Exception:
        return None


def check_promotion_mae_gate(
    client,
    challenger_run_id: str | None,
    champion_run_id: str | None,
    max_ratio: float,
) -> tuple[bool, str]:
    challenger_mae = run_test_mae(client, challenger_run_id)
    champion_mae = run_test_mae(client, champion_run_id)
    if challenger_mae is None or champion_mae is None:
        return True, "skipped (missing test_mae on run)"
    ratio = challenger_mae / champion_mae if champion_mae else float("inf")
    if ratio > max_ratio:
        return (
            False,
            f"challenger test_mae {challenger_mae:.0f} exceeds "
            f"{max_ratio:.0%} of champion MAE {champion_mae:.0f} (ratio={ratio:.2f})",
        )
    return True, f"ratio={ratio:.2f}"


def wait_for_endpoint(deploy_mod, host: str, token: str, endpoint: str) -> bool:
    import time

    for attempt in range(60):
        try:
            status_resp = deploy_mod.curl_json(
                "GET",
                f"{host}/api/2.0/serving-endpoints/{endpoint}",
                token,
            )
        except RuntimeError as exc:
            print(f"    Status check failed: {exc}", file=sys.stderr)
            time.sleep(15)
            continue

        state = status_resp.get("state") or {}
        ready = state.get("ready", "UNKNOWN")
        updating = state.get("config_update", "UNKNOWN")
        pending = status_resp.get("pending_config", {})
        entities = pending.get("served_entities") or status_resp.get("config", {}).get(
            "served_entities",
            [],
        )
        deployment = "n/a"
        if entities:
            entity_state = entities[0].get("state", {})
            deployment = entity_state.get("deployment", "n/a")
            message = entity_state.get("deployment_state_message", "")
            if message:
                deployment = f"{deployment}: {message[:120]}"

        print(f"    ready={ready}, config_update={updating}, deployment={deployment}")
        if updating == "UPDATE_FAILED":
            return False
        if ready == "READY" and updating in ("NOT_UPDATING", None, ""):
            return True
        time.sleep(15)

    print("    WARN: Endpoint not READY yet — check Serving UI in Databricks", file=sys.stderr)
    return False


def deploy_endpoint(
    deploy_mod,
    host: str,
    token: str,
    endpoint: str,
    model_name: str,
    version: str,
) -> None:
    served_entity = {
        "entity_name": model_name,
        "entity_version": version,
        "workload_size": "Small",
        "scale_to_zero_enabled": True,
    }
    create_body = {
        "name": endpoint,
        "config": {"served_entities": [served_entity]},
    }
    update_body = {"served_entities": [served_entity]}

    if deploy_mod.endpoint_exists(host, token, endpoint):
        print(f"    Endpoint exists — updating to version {version}")
        deploy_mod.curl_json(
            "PUT",
            f"{host}/api/2.0/serving-endpoints/{endpoint}/config",
            token,
            update_body,
        )
        return

    try:
        deploy_mod.curl_json(
            "POST",
            f"{host}/api/2.0/serving-endpoints",
            token,
            create_body,
        )
        print(f"    Endpoint creation started: {endpoint}")
    except RuntimeError as exc:
        err = str(exc)
        if "RESOURCE_ALREADY_EXISTS" in err or "already exists" in err.lower():
            deploy_mod.curl_json(
                "PUT",
                f"{host}/api/2.0/serving-endpoints/{endpoint}/config",
                token,
                update_body,
            )
            print("    Endpoint config updated")
        elif "RESOURCE_EXHAUSTED" in err:
            deploy_mod.print_endpoint_limit_help(host, token)
            raise
        else:
            raise


def write_promotion_record(root: Path, record: dict) -> Path:
    out_dir = root / "ml" / "artifacts" / "promotions"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "last-promotion.json"
    out_path.write_text(json.dumps(record, indent=2))
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Promote staging challenger to production champion")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be promoted without making changes",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    load_env(root / ".env")

    if not args.dry_run and os.environ.get("CONFIRM_PROMOTE") != "yes":
        print("ERROR: Refusing to promote without confirmation.", file=sys.stderr)
        print("Run: CONFIRM_PROMOTE=yes make promote-to-production", file=sys.stderr)
        return 1

    host = os.environ.get("DATABRICKS_HOST", "").rstrip("/")
    token = os.environ.get("DATABRICKS_TOKEN", "")
    schema = os.environ.get("DATABRICKS_SCHEMA", "gold")
    staging_catalog = os.environ.get("STAGING_CATALOG", "house_price_staging")
    prod_catalog = os.environ.get("PROD_CATALOG", "house_price_prod")
    source_alias = os.environ.get("PROMOTE_FROM_ALIAS", "challenger")
    target_alias = os.environ.get("PROMOTE_TO_ALIAS", "champion")
    rollback_alias = os.environ.get("PROMOTE_ROLLBACK_ALIAS", "previous_champion")
    prod_endpoint = os.environ.get(
        "PROD_SERVING_ENDPOINT",
        os.environ.get("DATABRICKS_SERVING_ENDPOINT_PROD", "house-price-serving-prod"),
    )

    if not host or not token:
        print("ERROR: Set DATABRICKS_HOST and DATABRICKS_TOKEN in .env", file=sys.stderr)
        return 1

    os.environ["DATABRICKS_HOST"] = host
    os.environ["DATABRICKS_TOKEN"] = token
    os.environ["MLFLOW_TRACKING_URI"] = "databricks"
    os.environ["MLFLOW_REGISTRY_URI"] = "databricks-uc"

    import mlflow
    from mlflow import MlflowClient
    from mlflow.exceptions import RestException

    sys.path.insert(0, str(root / "ml" / "src"))
    from house_price_ml.evaluation.gate_config import load_quality_gates

    mlflow.set_tracking_uri("databricks")
    mlflow.set_registry_uri("databricks-uc")

    staging_model = f"{staging_catalog}.{schema}.house_price_model"
    prod_model = f"{prod_catalog}.{schema}.house_price_model"
    client = MlflowClient(registry_uri="databricks-uc")

    print("==> 1/5 Resolve staging candidate")
    try:
        candidate = client.get_model_version_by_alias(staging_model, source_alias)
    except RestException as exc:
        print(f"ERROR: No '{source_alias}' on {staging_model}: {exc}", file=sys.stderr)
        print("Deploy a challenger to staging first: make deploy-serving", file=sys.stderr)
        return 1

    source_version = str(candidate.version)
    source_uri = f"models:/{staging_model}/{source_version}"
    source_tags = version_tags(client, staging_model, source_version)
    source_run_id = version_run_id(client, staging_model, source_version)
    print(f"    {staging_model} @{source_alias} -> version {source_version}")
    print(f"    source_uri: {source_uri}")
    if source_tags:
        print(f"    tags: {json.dumps(source_tags)}")

    previous_prod_version: str | None = None
    champion_run_id: str | None = None
    try:
        current_champion = client.get_model_version_by_alias(prod_model, target_alias)
        previous_prod_version = str(current_champion.version)
        champion_run_id = version_run_id(client, prod_model, previous_prod_version)
    except RestException:
        pass

    promotion_gates = load_quality_gates()
    mae_ok, mae_detail = check_promotion_mae_gate(
        client,
        source_run_id,
        champion_run_id,
        promotion_gates.promotion.max_mae_vs_champion_ratio,
    )
    if not mae_ok:
        print(f"ERROR: Promotion MAE gate failed — {mae_detail}", file=sys.stderr)
        return 1
    print(f"    MAE promotion gate: {mae_detail}")

    if args.dry_run:
        print("")
        print("DRY RUN — would:")
        print(f"  • Copy {source_uri} -> {prod_model}")
        if previous_prod_version:
            print(f"  • Set {rollback_alias} <- version {previous_prod_version}")
        print(f"  • Set {target_alias} <- copied version")
        print(f"  • Update endpoint {prod_endpoint}")
        return 0

    deploy_mod = load_deploy_serving_module()

    print("==> 2/5 Copy model version into production registry")
    try:
        registered = mlflow.register_model(model_uri=source_uri, name=prod_model)
        prod_version = str(registered.version)
    except Exception as exc:
        print(f"ERROR: Failed to register model in production: {exc}", file=sys.stderr)
        return 1
    print(f"    Registered {prod_model} version {prod_version} (from staging v{source_version})")

    print(f"==> 3/5 Update production aliases on {prod_model}")
    if previous_prod_version:
        client.set_registered_model_alias(prod_model, rollback_alias, previous_prod_version)
        print(f"    {rollback_alias} <- version {previous_prod_version}")
    else:
        print(f"    No existing {target_alias} — skip {rollback_alias}")
    client.set_registered_model_alias(prod_model, target_alias, prod_version)
    print(f"    {target_alias} <- version {prod_version}")

    print(f"==> 4/5 Deploy production endpoint: {prod_endpoint}")
    previous_served_version = deploy_mod._get_current_served_version(
        host, token, prod_endpoint, prod_model
    )
    if previous_served_version:
        print(f"    Rollback target (previous served version): {previous_served_version}")
    try:
        deploy_endpoint(deploy_mod, host, token, prod_endpoint, prod_model, prod_version)
    except RuntimeError as exc:
        print(f"ERROR: Endpoint deploy failed: {exc}", file=sys.stderr)
        return 1

    print("==> 5/5 Wait for endpoint READY (may take 5–15 min)...")
    wait_for_endpoint(deploy_mod, host, token, prod_endpoint)

    print("==> 6/6 Verify live inference")
    verify_script = Path(__file__).resolve().parent / "verify-inference.py"
    verify = subprocess.run(
        [
            sys.executable,
            str(verify_script),
            "--profile",
            "production",
            "--expected-version",
            prod_version,
            "--skip-e2e",
        ],
        check=False,
    )
    if verify.returncode != 0:
        print("ERROR: Production inference verification failed", file=sys.stderr)
        rolled_back = deploy_mod._rollback_after_verify_failure(
            host,
            token,
            endpoint=prod_endpoint,
            model_name=prod_model,
            failed_version=prod_version,
            previous_served_version=previous_served_version,
            alias=target_alias,
            profile="production",
            alias_version=previous_prod_version,
        )
        if rolled_back:
            print(
                "Promotion failed verification; production endpoint and @champion were rolled back.",
                file=sys.stderr,
            )
        else:
            print(
                "Promotion failed verification and rollback could not restore production serving.",
                file=sys.stderr,
            )
        return 1

    record = {
        "promoted_at": datetime.now(timezone.utc).isoformat(),
        "source": {
            "catalog": staging_catalog,
            "model_name": staging_model,
            "alias": source_alias,
            "version": source_version,
            "model_uri": source_uri,
            "tags": source_tags,
        },
        "target": {
            "catalog": prod_catalog,
            "model_name": prod_model,
            "alias": target_alias,
            "version": prod_version,
            "endpoint": prod_endpoint,
        },
        "rollback": {
            "alias": rollback_alias,
            "version": previous_prod_version,
        },
    }
    record_path = write_promotion_record(root, record)

    print("")
    print("Promotion complete.")
    print(f"  Staging:  {staging_model} @{source_alias} = v{source_version}")
    print(f"  Production: {prod_model} @{target_alias} = v{prod_version}")
    print(f"  Endpoint: {prod_endpoint}")
    print(f"  Record:   {record_path}")
    print("")
    print("Verify: curl https://pb-assessment.netlify.app/api/monitoring")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
