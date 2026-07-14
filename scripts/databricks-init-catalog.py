#!/usr/bin/env python3
"""Apply medallion DDL to Unity Catalog via Databricks SQL API (uses curl for SSL compatibility)."""
from __future__ import annotations

import json
import os
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


def split_statements(sql: str) -> list[str]:
    lines = []
    for line in sql.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("--"):
            continue
        lines.append(line)
    blob = "\n".join(lines)
    parts = [p.strip() for p in blob.split(";") if p.strip()]
    return [p + ";" for p in parts]


def execute(host: str, token: str, warehouse_id: str, statement: str) -> dict:
    payload = json.dumps(
        {"warehouse_id": warehouse_id, "statement": statement, "wait_timeout": "50s"}
    )
    result = subprocess.run(
        [
            "curl",
            "-sS",
            "-X",
            "POST",
            f"{host.rstrip('/')}/api/2.0/sql/statements",
            "-H",
            f"Authorization: Bearer {token}",
            "-H",
            "Content-Type: application/json",
            "-d",
            payload,
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"curl failed: {result.stderr or result.stdout}")
    return json.loads(result.stdout)


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    load_env(root / ".env")

    catalog = sys.argv[1] if len(sys.argv) > 1 else "house_price_staging"
    host = os.environ.get("DATABRICKS_HOST", "")
    token = os.environ.get("DATABRICKS_TOKEN", "")
    warehouse_id = os.environ.get("DATABRICKS_SQL_WAREHOUSE_ID", "")

    if not all([host, token, warehouse_id]):
        print("ERROR: Set DATABRICKS_HOST, DATABRICKS_TOKEN, DATABRICKS_SQL_WAREHOUSE_ID in .env", file=sys.stderr)
        return 1

    sql_dir = root / "databricks" / "sql"
    files = sorted(sql_dir.glob("[0-9]*.sql"))

    print(f"Creating catalog and tables in: {catalog}")
    for sql_file in files:
        print(f"  -> {sql_file.name}")
        content = sql_file.read_text().replace("${catalog}", catalog)
        for stmt in split_statements(content):
            preview = stmt.splitlines()[0][:70]
            print(f"      {preview}...")
            try:
                result = execute(host, token, warehouse_id, stmt)
            except (RuntimeError, json.JSONDecodeError) as e:
                print(f"    FAILED: {e}", file=sys.stderr)
                return 1

            state = result.get("status", {}).get("state", "UNKNOWN")
            if state != "SUCCEEDED":
                error_msg = result.get("status", {}).get("error", {}).get("message", "")
                if sql_file.name.startswith("04_") and "COLUMN_ALREADY_EXISTS" in error_msg:
                    print("    SKIP (column already exists)")
                    continue
                print(f"    FAILED (state={state})", file=sys.stderr)
                print(json.dumps(result, indent=2), file=sys.stderr)
                return 1
            print("    OK")

    print("Done. Verify with: make verify-databricks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
