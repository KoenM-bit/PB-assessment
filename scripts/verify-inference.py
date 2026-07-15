#!/usr/bin/env python3
"""Verify live inference after promoting and deploying a model to serving."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

SAMPLE_PAYLOAD = {
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

FALLBACK_VERSIONS = {"baseline", "mock-v1", "unknown"}


def load_env(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


def profile_config(profile: str) -> dict[str, str]:
    if profile == "staging":
        return {
            "catalog": os.environ.get("DATABRICKS_CATALOG", "house_price_staging"),
            "endpoint": os.environ.get("DATABRICKS_SERVING_ENDPOINT", "house-price-serving"),
            "alias": os.environ.get("MODEL_ALIAS", "challenger"),
            "e2e_url": os.environ.get(
                "E2E_BASE_URL",
                os.environ.get("STAGING_URL", "https://staging--pb-assessment.netlify.app"),
            ),
        }
    if profile in {"production", "prod"}:
        return {
            "catalog": os.environ.get("DATABRICKS_CATALOG", "house_price_prod"),
            "endpoint": os.environ.get(
                "DATABRICKS_SERVING_ENDPOINT",
                os.environ.get("DATABRICKS_SERVING_ENDPOINT_PROD", "house-price-serving-prod"),
            ),
            "alias": os.environ.get("MODEL_ALIAS", "champion"),
            "e2e_url": os.environ.get("E2E_BASE_URL", os.environ.get("PRODUCTION_URL", "")),
        }
    raise ValueError(f"Unknown profile: {profile}")


def databricks_json(method: str, url: str, token: str, body: dict | None = None) -> dict:
    data = None
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    if body is not None:
        data = json.dumps(body).encode()
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=90) as response:
            payload = response.read().decode()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode()
        raise RuntimeError(f"HTTP {exc.code}: {detail[:300]}") from exc
    if not payload.strip():
        return {}
    return json.loads(payload)


def resolve_alias_version(catalog: str, schema: str, alias: str) -> str:
    os.environ.setdefault("MLFLOW_TRACKING_URI", "databricks")
    os.environ.setdefault("MLFLOW_REGISTRY_URI", "databricks-uc")
    import mlflow
    from mlflow import MlflowClient

    mlflow.set_tracking_uri("databricks")
    mlflow.set_registry_uri("databricks-uc")
    model_name = f"{catalog}.{schema}.house_price_model"
    resolved = MlflowClient(registry_uri="databricks-uc").get_model_version_by_alias(
        model_name, alias
    )
    return str(resolved.version)


def endpoint_serves_version(
    host: str, token: str, endpoint: str, model_name: str, version: str
) -> bool:
    status = databricks_json("GET", f"{host}/api/2.0/serving-endpoints/{endpoint}", token)
    state = status.get("state") or {}
    if state.get("ready") != "READY":
        return False
    entities = status.get("config", {}).get("served_entities") or []
    return any(
        entity.get("entity_name") == model_name
        and str(entity.get("entity_version")) == str(version)
        for entity in entities
    )


def verify_serving_endpoint(
    host: str,
    token: str,
    endpoint: str,
    *,
    expected_version: str | None,
    model_name: str,
) -> dict:
    print(f"==> 1/2 Databricks serving smoke: {endpoint}")
    if expected_version:
        if not endpoint_serves_version(host, token, endpoint, model_name, expected_version):
            raise RuntimeError(
                f"Endpoint '{endpoint}' is not serving {model_name} version {expected_version}"
            )
        print(f"    Endpoint serves expected version {expected_version}")

    result = databricks_json(
        "POST",
        f"{host}/serving-endpoints/{endpoint}/invocations",
        token,
        {"dataframe_records": [SAMPLE_PAYLOAD]},
    )
    rows = result.get("predictions", result)
    row = rows[0] if isinstance(rows, list) else rows
    if isinstance(row, dict):
        price = row.get("predicted_price", row)
        model_version = str(
            result.get("model_version")
            or result.get("databricks_model_version")
            or row.get("model_version")
            or expected_version
            or ""
        )
    else:
        price = row
        model_version = str(expected_version or "")

    price_value = float(price)
    if price_value <= 0:
        raise RuntimeError(f"Invalid predicted_price from serving: {price_value}")
    if model_version.lower() in FALLBACK_VERSIONS:
        raise RuntimeError(f"Serving returned fallback-like model_version={model_version!r}")

    print(f"    predicted_price={price_value:.0f}, model_version={model_version or 'n/a'}")
    return {"predicted_price": price_value, "model_version": model_version}


def verify_staging_e2e(e2e_url: str, *, expected_alias: str) -> None:
    print(f"==> 2/2 Staging API inference smoke: {e2e_url}")
    env = os.environ.copy()
    env["E2E_BASE_URL"] = e2e_url.rstrip("/")
    env["E2E_REQUIRE_LIVE_SERVING"] = "true"
    env["E2E_EXPECTED_MODEL_ALIAS"] = expected_alias
    root = Path(__file__).resolve().parents[1]
    subprocess.run(
        [sys.executable, "-m", "pytest", "tests/e2e/test_serving_promote.py", "-v"],
        cwd=root,
        env=env,
        check=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify inference after model promotion/deploy.")
    parser.add_argument("--profile", choices=["staging", "production", "prod"], default="staging")
    parser.add_argument("--skip-e2e", action="store_true", help="Only hit Databricks serving.")
    parser.add_argument("--expected-version", default=None, help="Registry version to assert.")
    args = parser.parse_args()

    profile = "production" if args.profile == "prod" else args.profile
    root = Path(__file__).resolve().parents[1]
    load_env(root / ".env")

    host = os.environ.get("DATABRICKS_HOST", "").rstrip("/")
    token = os.environ.get("DATABRICKS_TOKEN", "")
    schema = os.environ.get("DATABRICKS_SCHEMA", "gold")
    cfg = profile_config(profile)

    if not host or not token:
        print("ERROR: Set DATABRICKS_HOST and DATABRICKS_TOKEN", file=sys.stderr)
        return 1

    catalog = cfg["catalog"]
    endpoint = cfg["endpoint"]
    alias = cfg["alias"]
    model_name = f"{catalog}.{schema}.house_price_model"
    expected_version = args.expected_version or resolve_alias_version(catalog, schema, alias)

    try:
        verify_serving_endpoint(
            host,
            token,
            endpoint,
            expected_version=expected_version,
            model_name=model_name,
        )
        if not args.skip_e2e and profile == "staging" and cfg["e2e_url"]:
            verify_staging_e2e(cfg["e2e_url"], expected_alias=alias)
        elif not args.skip_e2e and profile == "staging":
            print("    SKIP E2E: set E2E_BASE_URL or STAGING_URL for Netlify API check")
    except (RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"ERROR: Inference verification failed: {exc}", file=sys.stderr)
        return 1

    print("")
    print("Inference verification passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
