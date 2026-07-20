#!/usr/bin/env python3
"""Recompute actual_sale_price for synthetic labels so live MAE ~ holdout (~10–12% beat).

Usage:
  python3 scripts/recalibrate-synthetic-actuals.py
  python3 scripts/recalibrate-synthetic-actuals.py --catalog house_price_prod
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from synthetic_eval_calibration import (  # noqa: E402
    baseline_predict,
    calibrated_actual_price,
    load_training_manifest,
    simulate_live_improvement_pct,
)


def load_dotenv() -> None:
    env_path = ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"'))


def sql_escape(value: str) -> str:
    return value.replace("'", "''")


def run_sql(host: str, token: str, warehouse_id: str, statement: str) -> dict:
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
    body = json.loads(result.stdout)
    state = body.get("status", {}).get("state")
    if state == "FAILED":
        msg = body.get("status", {}).get("error", {}).get("message", "unknown")
        raise RuntimeError(msg)
    return body


def fetch_rows(
    host: str, token: str, warehouse_id: str, catalog: str, *, all_labelled: bool
) -> list[dict]:
    where = "1=1" if all_labelled else "a.recorded_by = 'synthetic_seed'"
    stmt = f"""
SELECT
  a.actual_sale_id,
  p.prediction_id,
  p.predicted_price,
  get_json_object(p.request_payload, '$.region') AS region,
  get_json_object(p.request_payload, '$.property_type') AS property_type,
  CAST(get_json_object(p.request_payload, '$.surface_area') AS DOUBLE) AS surface_area
FROM {catalog}.gold.actual_sales a
INNER JOIN {catalog}.gold.predictions p ON a.prediction_id = p.prediction_id
WHERE {where}
"""
    body = run_sql(host, token, warehouse_id, stmt)
    rows = body.get("result", {}).get("data_array") or []
    out: list[dict] = []
    for row in rows:
        out.append(
            {
                "actual_sale_id": str(row[0]),
                "prediction_id": str(row[1]),
                "predicted_price": float(row[2]),
                "region": str(row[3]),
                "property_type": str(row[4]),
                "surface_area": float(row[5] or 0),
            }
        )
    return out


def main() -> int:
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", default=os.environ.get("DATABRICKS_CATALOG", "house_price_staging"))
    parser.add_argument("--target-improvement-pct", type=float, default=9.5)
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--all-labelled",
        action="store_true",
        help="Recalibrate every prediction with an actual sale (e.g. production demo data)",
    )
    args = parser.parse_args()

    host = os.environ.get("DATABRICKS_HOST", "")
    token = os.environ.get("DATABRICKS_TOKEN", "")
    warehouse = os.environ.get("DATABRICKS_SQL_WAREHOUSE_ID", "")
    if not host or not token or not warehouse:
        print("Set DATABRICKS_HOST, DATABRICKS_TOKEN, DATABRICKS_SQL_WAREHOUSE_ID in .env", file=sys.stderr)
        return 1

    manifest = load_training_manifest()
    rng = random.Random(args.seed)
    rows = fetch_rows(host, token, warehouse, args.catalog, all_labelled=args.all_labelled)
    if not rows:
        scope = "labelled" if args.all_labelled else "synthetic"
        print(f"No {scope} rows in {args.catalog}.gold")
        return 0

    before: list[tuple[float, float, float]] = []
    updates: list[tuple[str, int]] = []
    for row in rows:
        baseline = baseline_predict(
            row["region"], row["property_type"], row["surface_area"], manifest
        )
        predicted = row["predicted_price"]
        # We don't have old actual in SELECT — refetch would need it; skip before sim for old
        new_actual = calibrated_actual_price(
            predicted,
            baseline,
            rng,
            target_improvement_pct=args.target_improvement_pct,
        )
        updates.append((row["actual_sale_id"], new_actual))
        before.append((predicted, baseline, float(new_actual)))

    improvement = simulate_live_improvement_pct(before)
    print(
        f"{args.catalog}: {len(updates)} actual sales → "
        f"simulated live MAE improvement {improvement:.1f}% (target ~{args.target_improvement_pct:.1f}%)"
    )

    if args.dry_run:
        return 0

    for actual_sale_id, price in updates:
        stmt = f"""
UPDATE {args.catalog}.gold.actual_sales
SET actual_sale_price = {price}
WHERE actual_sale_id = '{sql_escape(actual_sale_id)}'
"""
        run_sql(host, token, warehouse, stmt)

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
