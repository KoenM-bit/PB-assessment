#!/usr/bin/env python3
"""Print Databricks Model Serving build/server logs for debugging deploy failures."""
from __future__ import annotations

import json
import os
import re
import subprocess
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


def curl_json(url: str, token: str) -> dict:
    result = subprocess.run(
        ["curl", "-sS", url, "-H", f"Authorization: Bearer {token}"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr or result.stdout)
    return json.loads(result.stdout or "{}")


def extract_errors(logs: str) -> list[str]:
    patterns = [
        r"missing_module[^\"]*\"([^\"]+)\"",
        r"No module named '([^']+)'",
        r"ModuleNotFoundError: (.+)",
        r"DATABRICKS_MODEL_LOAD_DIAGNOSTIC_V1 ({.+})",
    ]
    found: list[str] = []
    for line in logs.splitlines():
        if re.search(r"(?i)error|traceback|failed|modulenotfound|diagnostic", line):
            found.append(line.strip())
        for pattern in patterns:
            match = re.search(pattern, line)
            if match:
                found.append(f"  -> {match.group(1)}")
    return found


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    load_env(root / ".env")

    host = os.environ.get("DATABRICKS_HOST", "").rstrip("/")
    token = os.environ.get("DATABRICKS_TOKEN", "")
    endpoint = os.environ.get("DATABRICKS_SERVING_ENDPOINT", "house-price-serving")
    served_model = sys.argv[1] if len(sys.argv) > 1 else None

    if not host or not token:
        print("Set DATABRICKS_HOST and DATABRICKS_TOKEN in .env", file=sys.stderr)
        return 1

    status = curl_json(f"{host}/api/2.0/serving-endpoints/{endpoint}", token)
    pending = status.get("pending_config") or status.get("config") or {}
    entities = pending.get("served_entities") or []
    if not served_model and entities:
        served_model = entities[0].get("name")
    config_version = pending.get("config_version", 0)

    if not served_model:
        print("No served model found on endpoint", file=sys.stderr)
        return 1

    print(f"Endpoint: {endpoint}")
    print(f"Served model: {served_model}")
    print(f"Config version: {config_version}")
    print("")

    for log_type in ("build-logs", "logs"):
        url = (
            f"{host}/api/2.0/serving-endpoints/{endpoint}/served-models/"
            f"{served_model}/{log_type}?config_version={config_version}"
        )
        data = curl_json(url, token)
        logs = data.get("logs", "")
        print(f"=== {log_type} (last 4000 chars) ===")
        print(logs[-4000:] if logs else data)
        errors = extract_errors(logs)
        if errors:
            print(f"\n=== parsed issues from {log_type} ===")
            for line in errors[-20:]:
                print(line)
        print("")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
