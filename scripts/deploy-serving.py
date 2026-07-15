#!/usr/bin/env python3
"""Register local MLflow model to Unity Catalog and create Model Serving endpoint."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
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


def curl_json(method: str, url: str, token: str, body: dict | None = None) -> dict:
    cmd = [
        "curl",
        "-sS",
        "-w",
        "\n__HTTP_CODE__:%{http_code}",
        "-X",
        method,
        url,
        "-H",
        f"Authorization: Bearer {token}",
        "-H",
        "Content-Type: application/json",
    ]
    if body is not None:
        cmd += ["-d", json.dumps(body)]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(result.stderr or result.stdout)
    raw = result.stdout
    if "\n__HTTP_CODE__:" in raw:
        payload, _, code_str = raw.rpartition("\n__HTTP_CODE__:")
        http_code = int(code_str.strip() or "0")
    else:
        payload, http_code = raw, 200
    if not payload.strip():
        if http_code >= 400:
            raise RuntimeError(f"HTTP {http_code}: empty response")
        return {}
    data = json.loads(payload)
    if http_code >= 400:
        message = data.get("message", payload)
        error_code = data.get("error_code", "HTTP_ERROR")
        raise RuntimeError(f"{error_code}: {message}")
    return data


def curl_status(method: str, url: str, token: str, body: dict | None = None) -> tuple[int, str]:
    cmd = ["curl", "-sS", "-w", "%{http_code}", "-o", "-", "-X", method, url, "-H", f"Authorization: Bearer {token}", "-H", "Content-Type: application/json"]
    if body is not None:
        cmd += ["-d", json.dumps(body)]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        return 0, result.stderr or result.stdout
    if len(result.stdout) < 3:
        return 0, result.stdout
    return int(result.stdout[-3:]), result.stdout[:-3]


def print_token_scope_help(missing: list[str]) -> None:
    scopes = ", ".join(missing)
    print("", file=sys.stderr)
    print(f"ERROR: Your DATABRICKS_TOKEN is missing scope(s): {scopes}", file=sys.stderr)
    print("", file=sys.stderr)
    print("Create a new Personal Access Token:", file=sys.stderr)
    print("  1. Databricks → click your username (top right) → Settings", file=sys.stderr)
    print("  2. Developer → Access tokens → Generate new token", file=sys.stderr)
    print("  3. Choose **All APIs** (recommended for local deploy).", file=sys.stderr)
    print("     Uploading local model files to Unity Catalog needs the all-apis scope.", file=sys.stderr)
    print("     SQL + MLflow + Serving scopes alone are not enough for make deploy-serving.", file=sys.stderr)
    print("  4. Copy the new token into .env as DATABRICKS_TOKEN", file=sys.stderr)
    print("  5. Re-run: make train && make deploy-serving", file=sys.stderr)
    print("", file=sys.stderr)
    print("Old tokens cannot be edited — you must generate a new one.", file=sys.stderr)


def endpoint_exists(host: str, token: str, endpoint: str) -> bool:
    code, body = curl_status("GET", f"{host}/api/2.0/serving-endpoints/{endpoint}", token)
    if code == 200:
        return True
    if code == 404 or "RESOURCE_DOES_NOT_EXIST" in body:
        return False
    return False


def print_endpoint_limit_help(host: str, token: str) -> None:
    print("", file=sys.stderr)
    print("ERROR: Databricks free tier limits custom serving endpoints (typically 1).", file=sys.stderr)
    print("", file=sys.stderr)
    print("You already have custom endpoints in this workspace. Options:", file=sys.stderr)
    try:
        listed = curl_json("GET", f"{host}/api/2.0/serving-endpoints", token)
        for item in listed.get("endpoints", []):
            name = item.get("name", "")
            if not name.startswith("databricks-"):
                ready = (item.get("state") or {}).get("ready", "?")
                print(f"  • {name} ({ready})", file=sys.stderr)
    except RuntimeError:
        pass
    print("", file=sys.stderr)
    print("Recommended for free tier:", file=sys.stderr)
    print("  1. Databricks → Serving → delete house-price-serving-prod (keep staging only)", file=sys.stderr)
    print("  2. Re-run: make deploy-serving", file=sys.stderr)
    print("", file=sys.stderr)
    print("Staging + production can share one endpoint on free tier; use separate", file=sys.stderr)
    print("catalogs/aliases in the app, and promote model versions instead.", file=sys.stderr)


def _print_serving_failure_logs(
    host: str, token: str, endpoint: str, served_model: str, config_version: int
) -> None:
    import re

    for log_type in ("logs", "build-logs"):
        try:
            data = curl_json(
                "GET",
                f"{host}/api/2.0/serving-endpoints/{endpoint}/served-models/"
                f"{served_model}/{log_type}?config_version={config_version}",
                token,
            )
        except RuntimeError:
            continue
        logs = data.get("logs", "")
        if not logs:
            continue
        print(f"    --- {log_type} ---", file=sys.stderr)
        for line in logs.splitlines():
            if re.search(
                r"(?i)modulenotfound|no module named|diagnostic_v1|failed to import|PhaseTaggedError",
                line,
            ):
                print(f"    {line.strip()[:300]}", file=sys.stderr)
        match = re.search(r"missing_module[^\"]*\"([^\"]+)\"", logs)
        if match:
            print(f"    MISSING MODULE: {match.group(1)}", file=sys.stderr)
        match = re.search(r"No module named '([^']+)'", logs)
        if match:
            print(f"    MISSING MODULE: {match.group(1)}", file=sys.stderr)


def check_token_scopes(host: str, token: str) -> list[str]:
    missing: list[str] = []
    checks = [
        ("all-apis", "POST", f"{host}/api/2.0/workspace/list", {"path": "/"}),
        ("sql", "POST", f"{host}/api/2.0/sql/statements", {"warehouse_id": os.environ.get("DATABRICKS_SQL_WAREHOUSE_ID", ""), "statement": "SELECT 1", "wait_timeout": "10s"}),
        ("mlflow", "GET", f"{host}/api/2.0/mlflow/registered-models/search?max_results=1", None),
        ("serving", "GET", f"{host}/api/2.0/serving-endpoints", None),
    ]
    for scope, method, url, body in checks:
        code, body_text = curl_status(method, url, token, body)
        if code == 403 and f"scopes: {scope}" in body_text:
            missing.append(scope)
        elif code == 0:
            missing.append(scope)
    return missing


def _endpoint_serves_version(
    host: str, token: str, endpoint: str, model_name: str, version: str
) -> bool:
    try:
        status_resp = curl_json("GET", f"{host}/api/2.0/serving-endpoints/{endpoint}", token)
    except RuntimeError:
        return False
    state = status_resp.get("state") or {}
    if state.get("ready") != "READY":
        return False
    entities = status_resp.get("config", {}).get("served_entities") or []
    return any(
        entity.get("entity_name") == model_name and str(entity.get("entity_version")) == str(version)
        for entity in entities
    )


def _get_current_served_version(
    host: str, token: str, endpoint: str, model_name: str
) -> str | None:
    if not endpoint_exists(host, token, endpoint):
        return None
    try:
        status_resp = curl_json("GET", f"{host}/api/2.0/serving-endpoints/{endpoint}", token)
    except RuntimeError:
        return None
    entities = status_resp.get("config", {}).get("served_entities") or []
    for entity in entities:
        if entity.get("entity_name") == model_name:
            version = entity.get("entity_version")
            if version is not None:
                return str(version)
    return None


def _apply_endpoint_config(
    host: str, token: str, endpoint: str, model_name: str, serve_version: str
) -> None:
    update_body = {
        "served_entities": [
            {
                "entity_name": model_name,
                "entity_version": serve_version,
                "workload_size": "Small",
                "scale_to_zero_enabled": True,
            }
        ]
    }
    curl_json(
        "PUT",
        f"{host}/api/2.0/serving-endpoints/{endpoint}/config",
        token,
        update_body,
    )


def _set_registry_alias(model_name: str, alias: str, version: str) -> None:
    os.environ.setdefault("MLFLOW_TRACKING_URI", "databricks")
    os.environ.setdefault("MLFLOW_REGISTRY_URI", "databricks-uc")
    import mlflow
    from mlflow import MlflowClient

    mlflow.set_tracking_uri("databricks")
    mlflow.set_registry_uri("databricks-uc")
    MlflowClient(registry_uri="databricks-uc").set_registered_model_alias(
        model_name, alias, str(version)
    )


def _wait_for_endpoint_ready(host: str, token: str, endpoint: str) -> bool:
    for _attempt in range(60):
        try:
            status_resp = curl_json("GET", f"{host}/api/2.0/serving-endpoints/{endpoint}", token)
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
            entity_name = entities[0].get("name") if entities else None
            config_version = pending.get("config_version") or status_resp.get("config", {}).get(
                "config_version",
                0,
            )
            if entity_name:
                print("", file=sys.stderr)
                print("    Deployment failed. Fetching serving logs...", file=sys.stderr)
                _print_serving_failure_logs(host, token, endpoint, entity_name, config_version)
                print("", file=sys.stderr)
                print("    Fix the issue, then: make train && make deploy-serving", file=sys.stderr)
                print("    Or inspect logs anytime: make fetch-serving-logs", file=sys.stderr)
            return False
        if ready == "READY" and updating in ("NOT_UPDATING", None, ""):
            return True
        time.sleep(15)

    print("    WARN: Endpoint not READY yet — check Serving UI in Databricks", file=sys.stderr)
    return False


def _run_inference_verification(
    profile: str, expected_version: str, *, skip_e2e: bool = False
) -> int:
    script = Path(__file__).resolve().parent / "verify-inference.py"
    cmd = [
        sys.executable,
        str(script),
        "--profile",
        profile,
        "--expected-version",
        expected_version,
    ]
    if skip_e2e:
        cmd.append("--skip-e2e")
    result = subprocess.run(cmd, env=os.environ.copy(), check=False)
    return result.returncode


def _rollback_after_verify_failure(
    host: str,
    token: str,
    *,
    endpoint: str,
    model_name: str,
    failed_version: str,
    previous_served_version: str | None,
    alias: str | None,
    profile: str,
    alias_version: str | None = None,
) -> bool:
    if not previous_served_version or previous_served_version == failed_version:
        print(
            "    No previous serving version available — rollback skipped.",
            file=sys.stderr,
        )
        return False

    restore_alias_version = alias_version or previous_served_version

    print("", file=sys.stderr)
    print(
        f"==> ROLLBACK: reverting failed deploy (v{failed_version}) "
        f"to previous version {previous_served_version}",
        file=sys.stderr,
    )
    try:
        _apply_endpoint_config(host, token, endpoint, model_name, previous_served_version)
        print(f"    Endpoint config rolled back to version {previous_served_version}")
        if not _wait_for_endpoint_ready(host, token, endpoint):
            print("    WARN: Rollback endpoint did not reach READY in time", file=sys.stderr)
        if alias:
            _set_registry_alias(model_name, alias, restore_alias_version)
            print(f"    Registry alias @{alias} restored to version {restore_alias_version}")
        if _run_inference_verification(profile, previous_served_version, skip_e2e=True) == 0:
            print(f"    Rollback verified — serving restored to version {previous_served_version}")
            return True
        print("    WARN: Rollback deployed but serving smoke still failing", file=sys.stderr)
    except Exception as exc:
        print(f"    ERROR: Rollback failed: {exc}", file=sys.stderr)
    return False


def deploy_endpoint(
    host: str,
    token: str,
    model_name: str,
    serve_version: str,
    endpoint: str,
    *,
    alias: str | None = None,
) -> int:
    previous_served_version = _get_current_served_version(host, token, endpoint, model_name)
    if previous_served_version:
        print(f"    Rollback target (previous served version): {previous_served_version}")

    print(f"==> 3/4 Create or update serving endpoint: {endpoint}")
    served_entity = {
        "entity_name": model_name,
        "entity_version": serve_version,
        "workload_size": "Small",
        "scale_to_zero_enabled": True,
    }
    create_body = {
        "name": endpoint,
        "config": {"served_entities": [served_entity]},
    }
    update_body = {"served_entities": [served_entity]}

    if endpoint_exists(host, token, endpoint):
        print(f"    Endpoint exists — updating to version {serve_version}")
        curl_json(
            "PUT",
            f"{host}/api/2.0/serving-endpoints/{endpoint}/config",
            token,
            update_body,
        )
        print("    Endpoint config updated")
    else:
        try:
            resp = curl_json("POST", f"{host}/api/2.0/serving-endpoints", token, create_body)
            print(f"    Endpoint creation started: {resp.get('name', endpoint)}")
        except RuntimeError as exc:
            err = str(exc)
            if "RESOURCE_ALREADY_EXISTS" in err or "already exists" in err.lower():
                curl_json(
                    "PUT",
                    f"{host}/api/2.0/serving-endpoints/{endpoint}/config",
                    token,
                    update_body,
                )
                print("    Endpoint config updated")
            elif "RESOURCE_EXHAUSTED" in err:
                print_endpoint_limit_help(host, token)
                return 1
            else:
                print(f"    Endpoint create failed: {err}", file=sys.stderr)
                return 1

    print("==> 4/4 Wait for endpoint to be READY (may take 5–15 min)...")
    if not _wait_for_endpoint_ready(host, token, endpoint):
        return 1

    profile = "production" if "prod" in endpoint else "staging"
    print("")
    print("==> 5/5 Verify live inference")
    if _run_inference_verification(profile, serve_version) != 0:
        rolled_back = _rollback_after_verify_failure(
            host,
            token,
            endpoint=endpoint,
            model_name=model_name,
            failed_version=serve_version,
            previous_served_version=previous_served_version,
            alias=alias,
            profile=profile,
        )
        if rolled_back:
            print(
                "Deploy failed verification; endpoint and alias were rolled back.",
                file=sys.stderr,
            )
        else:
            print(
                "Deploy failed verification and rollback could not restore serving.",
                file=sys.stderr,
            )
        return 1

    print("")
    print("Done. Serving deploy + inference verification passed.")
    return 0


def deploy_from_registry(
    host: str,
    token: str,
    catalog: str,
    schema: str,
    endpoint: str,
    alias: str,
) -> int:
    """Update serving endpoint to the model version behind a registry alias."""
    import mlflow
    from mlflow import MlflowClient

    model_name = f"{catalog}.{schema}.house_price_model"
    print(f"==> 1/2 Resolve registry alias: {model_name}@{alias}")
    mlflow.set_tracking_uri("databricks")
    mlflow.set_registry_uri("databricks-uc")
    client = MlflowClient(registry_uri="databricks-uc")
    try:
        resolved = client.get_model_version_by_alias(model_name, alias)
        serve_version = str(resolved.version)
    except Exception as exc:
        print(f"    Could not resolve alias '{alias}': {exc}", file=sys.stderr)
        print("    Run the train job or pipeline first.", file=sys.stderr)
        return 1
    print(f"    {alias} -> version {serve_version}")

    if _endpoint_serves_version(host, token, endpoint, model_name, serve_version):
        print(f"    Endpoint '{endpoint}' already serves version {serve_version} — no update needed")
        profile = "production" if "prod" in endpoint else "staging"
        print("")
        print("==> Verify live inference")
        return _run_inference_verification(profile, serve_version)

    return deploy_endpoint(
        host, token, model_name, serve_version, endpoint, alias=alias
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Register model and deploy Databricks serving endpoint.")
    parser.add_argument(
        "--from-registry",
        action="store_true",
        help="Skip local artifact upload; deploy the version behind MODEL_ALIAS in Unity Catalog.",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    load_env(root / ".env")

    host = os.environ.get("DATABRICKS_HOST", "").rstrip("/")
    token = os.environ.get("DATABRICKS_TOKEN", "")
    catalog = os.environ.get("DATABRICKS_CATALOG", "house_price_staging")
    schema = os.environ.get("DATABRICKS_SCHEMA", "gold")
    endpoint = os.environ.get("DATABRICKS_SERVING_ENDPOINT", "house-price-serving")
    alias = os.environ.get("MODEL_ALIAS", "challenger")

    if not host or not token:
        print("ERROR: Set DATABRICKS_HOST and DATABRICKS_TOKEN in .env", file=sys.stderr)
        return 1

    print("==> 0/4 Check token scopes")
    missing_scopes = check_token_scopes(host, token)
    if missing_scopes:
        print_token_scope_help(missing_scopes)
        return 1
    print("    Token scopes OK (all-apis, sql, mlflow, serving)")

    if args.from_registry:
        return deploy_from_registry(host, token, catalog, schema, endpoint, alias)

    model_uri_path = root / "ml" / "artifacts" / "model" / "mlflow_model"
    if not model_uri_path.exists():
        print("Model artifact not found. Run: make train", file=sys.stderr)
        return 1

    mlmodel = model_uri_path / "MLmodel"
    if mlmodel.exists() and "signature:" not in mlmodel.read_text():
        print(
            "Model artifact has no MLflow signature (required for Unity Catalog).",
            file=sys.stderr,
        )
        print("Re-train, then deploy again: make train && make deploy-serving", file=sys.stderr)
        return 1

    model_name = f"{catalog}.{schema}.house_price_model"
    print(f"==> 1/4 Register model: {model_name}")

    os.environ["DATABRICKS_HOST"] = host
    os.environ["DATABRICKS_TOKEN"] = token
    os.environ["MLFLOW_TRACKING_URI"] = "databricks"
    os.environ["MLFLOW_REGISTRY_URI"] = "databricks-uc"

    import mlflow
    from mlflow import MlflowClient

    mlflow.set_tracking_uri("databricks")
    mlflow.set_registry_uri("databricks-uc")

    model_uri = model_uri_path.resolve().as_uri()
    try:
        registered = mlflow.register_model(model_uri=model_uri, name=model_name)
        version = str(registered.version)
    except Exception as exc:
        err = str(exc)
        print(f"    Register failed: {err}", file=sys.stderr)
        if "scopes: mlflow" in err:
            print_token_scope_help(["mlflow"])
        elif "scopes: all-apis" in err:
            print_token_scope_help(["all-apis"])
        elif "signature" in err.lower():
            print("Re-train the model so it includes an MLflow signature:", file=sys.stderr)
            print("  make train && make deploy-serving", file=sys.stderr)
        return 1
    print(f"    Registered version: {version}")

    print(f"==> 2/4 Set alias '{alias}' on version {version}")
    try:
        client = MlflowClient(registry_uri="databricks-uc")
        client.set_registered_model_alias(model_name, alias, version)
        resolved = client.get_model_version_by_alias(model_name, alias)
        serve_version = str(resolved.version)
    except Exception as exc:
        print(f"    Alias failed: {exc}", file=sys.stderr)
        return 1
    print(f"    Alias '{alias}' -> version {serve_version}")

    return deploy_endpoint(
        host, token, model_name, serve_version, endpoint, alias=alias
    )


if __name__ == "__main__":
    raise SystemExit(main())
