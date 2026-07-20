#!/usr/bin/env python3
"""Seed gold.predictions + gold.actual_sales with synthetic time-series data.

Spreads requests across past days and the next week (for monitoring trends).
Requires .env with Databricks SQL credentials (same as cleanup script).

Usage:
  python3 scripts/seed-synthetic-history.py
  python3 scripts/seed-synthetic-history.py --days-past 14 --days-future 7 --per-day 3
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import uuid
from datetime import date, datetime, timedelta, timezone
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from synthetic_eval_calibration import (  # noqa: E402
    baseline_predict,
    calibrated_actual_price,
    load_training_manifest,
)

REGIONS = [
    "Amsterdam",
    "Rotterdam",
    "Utrecht",
    "The Hague",
    "Eindhoven",
    "Groningen",
    "Maastricht",
    "Nijmegen",
]
PROPERTY_TYPES = ["apartment", "terraced_house", "semi_detached", "detached", "bungalow"]
ENERGY = ["A", "B", "C", "D", "E"]
REGION_PSM = {
    "Amsterdam": 6200,
    "Rotterdam": 3800,
    "Utrecht": 4500,
    "The Hague": 4200,
    "Eindhoven": 3400,
    "Groningen": 3100,
    "Maastricht": 3600,
    "Nijmegen": 3300,
}
REGION_COORDS = {
    "Amsterdam": (52.3676, 4.9041),
    "Rotterdam": (51.9244, 4.4777),
    "Utrecht": (52.0907, 5.1214),
    "The Hague": (52.0705, 4.3007),
    "Eindhoven": (51.4416, 5.4697),
    "Groningen": (53.2194, 6.5665),
    "Maastricht": (50.8514, 5.6910),
    "Nijmegen": (51.8426, 5.8528),
}


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


def run_sql(host: str, token: str, warehouse_id: str, statement: str) -> None:
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


def build_property_key(address: str, postcode: str, region: str) -> str:
    parts = [address.strip().lower(), postcode.strip().lower(), region.strip().lower()]
    return "|".join(parts).replace("  ", " ")


def main() -> int:
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--days-past", type=int, default=14)
    parser.add_argument("--days-future", type=int, default=7)
    parser.add_argument("--per-day", type=int, default=3)
    parser.add_argument("--label-rate", type=float, default=0.55, help="Share with actual sale")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    host = os.environ.get("DATABRICKS_HOST", "")
    token = os.environ.get("DATABRICKS_TOKEN", "")
    warehouse = os.environ.get("DATABRICKS_SQL_WAREHOUSE_ID", "")
    catalog = os.environ.get("DATABRICKS_CATALOG", "house_price_staging")
    if not host or not token or not warehouse:
        print("Set DATABRICKS_HOST, DATABRICKS_TOKEN, DATABRICKS_SQL_WAREHOUSE_ID in .env", file=sys.stderr)
        return 1

    rng = random.Random(args.seed)
    manifest = load_training_manifest()
    today = date.today()
    app_env = os.environ.get("APP_ENV", "staging")

    predictions: list[dict] = []
    actuals: list[dict] = []

    seq = 0
    for day_offset in range(-args.days_past, args.days_future + 1):
        day = today + timedelta(days=day_offset)
        for _ in range(args.per_day):
            seq += 1
            region = REGIONS[(seq + day_offset) % len(REGIONS)]
            ptype = PROPERTY_TYPES[(seq * 3 + day_offset) % len(PROPERTY_TYPES)]
            surface = rng.randint(55, 195)
            if seq % 9 == 0:
                surface = 230
            if seq % 13 == 0:
                surface = 48
            rooms = rng.randint(3, 7)
            bedrooms = max(1, rooms - rng.randint(1, 2))
            build_year = rng.randint(1965, 2018)
            energy = rng.choice(ENERGY)
            lat, lon = REGION_COORDS[region]
            address = f"Synthetisch Laan {seq}"
            postcode = "3512 JC"
            payload = {
                "address": address,
                "postcode": postcode,
                "surface_area": surface,
                "number_of_rooms": rooms,
                "number_of_bedrooms": bedrooms,
                "build_year": build_year,
                "energy_label": energy,
                "property_type": ptype,
                "garden": seq % 2 == 0,
                "region": region,
                "latitude": lat,
                "longitude": lon,
                "prediction_date": day.isoformat(),
            }
            psm = REGION_PSM[region]
            predicted = round(psm * surface * rng.uniform(0.96, 1.04))
            hour = rng.randint(9, 18)
            minute = rng.randint(0, 59)
            ts = datetime(
                day.year, day.month, day.day, hour, minute, tzinfo=timezone.utc
            )
            prediction_id = str(uuid.uuid4())
            latency = rng.randint(180, 4200)
            predictions.append(
                {
                    "prediction_id": prediction_id,
                    "address": address,
                    "postcode": postcode,
                    "property_key": build_property_key(address, postcode, region),
                    "request_payload": json.dumps(payload),
                    "predicted_price": predicted,
                    "prediction_timestamp": ts.isoformat().replace("+00:00", "Z"),
                    "serving_latency_ms": latency,
                    "region": region,
                    "property_type": ptype,
                    "surface_area": surface,
                    "day": day,
                }
            )

            if rng.random() < args.label_rate:
                sale_day = day + timedelta(days=rng.randint(14, 45))
                if day_offset >= 0:
                    sale_day = day + timedelta(days=rng.randint(3, 12))
                baseline = baseline_predict(region, ptype, surface, manifest)
                actual_price = calibrated_actual_price(
                    float(predicted),
                    baseline,
                    rng,
                    target_improvement_pct=9.5,
                )
                actuals.append(
                    {
                        "actual_sale_id": str(uuid.uuid4()),
                        "prediction_id": prediction_id,
                        "actual_sale_price": actual_price,
                        "sale_date": sale_day.isoformat(),
                        "recorded_at": (ts + timedelta(hours=1)).isoformat().replace("+00:00", "Z"),
                    }
                )

    print(f"Inserting {len(predictions)} predictions and {len(actuals)} actual sales into {catalog}.gold …")

    for row in predictions:
        stmt = f"""
INSERT INTO {catalog}.gold.predictions (
  prediction_id, listing_id, address, postcode, property_key, request_payload,
  predicted_price, model_name, model_version, model_alias, prediction_timestamp,
  app_env, serving_latency_ms, warnings, is_fallback
) VALUES (
  '{sql_escape(row["prediction_id"])}',
  NULL,
  '{sql_escape(row["address"])}',
  '{sql_escape(row["postcode"])}',
  '{sql_escape(row["property_key"])}',
  '{sql_escape(row["request_payload"])}',
  {row["predicted_price"]},
  'house_price_model',
  '11',
  'challenger',
  TIMESTAMP '{row["prediction_timestamp"].replace("T", " ").replace("Z", "")}',
  '{sql_escape(app_env)}',
  {row["serving_latency_ms"]},
  array(),
  false
)
"""
        run_sql(host, token, warehouse, stmt)

    for row in actuals:
        stmt = f"""
INSERT INTO {catalog}.gold.actual_sales (
  actual_sale_id, prediction_id, listing_id, actual_sale_price, sale_date, recorded_at, recorded_by
) VALUES (
  '{sql_escape(row["actual_sale_id"])}',
  '{sql_escape(row["prediction_id"])}',
  NULL,
  {row["actual_sale_price"]},
  DATE '{row["sale_date"]}',
  TIMESTAMP '{row["recorded_at"].replace("T", " ").replace("Z", "")}',
  'synthetic_seed'
)
"""
        run_sql(host, token, warehouse, stmt)

    by_day: dict[str, int] = {}
    for row in predictions:
        by_day[str(row["day"])] = by_day.get(str(row["day"]), 0) + 1
    print("Predictions per day:")
    for d in sorted(by_day):
        print(f"  {d}: {by_day[d]}")
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
