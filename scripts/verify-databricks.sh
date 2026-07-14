#!/usr/bin/env bash
# Verify Databricks SQL Warehouse + Model Serving connectivity.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

: "${DATABRICKS_HOST:?Set DATABRICKS_HOST in .env}"
: "${DATABRICKS_TOKEN:?Set DATABRICKS_TOKEN in .env}"
: "${DATABRICKS_SQL_WAREHOUSE_ID:?Set DATABRICKS_SQL_WAREHOUSE_ID in .env}"

# Normalize host (no trailing slash)
DATABRICKS_HOST="${DATABRICKS_HOST%/}"

ENDPOINT="${DATABRICKS_SERVING_ENDPOINT:-house-price-serving}"
CATALOG="${DATABRICKS_CATALOG:-house_price_staging}"

# Warn if warehouse ID looks like workspace ID (long numeric) vs real warehouse ID (hex)
if [[ "${DATABRICKS_SQL_WAREHOUSE_ID}" =~ ^[0-9]{10,}$ ]]; then
  echo "WARN: DATABRICKS_SQL_WAREHOUSE_ID looks like a workspace ID, not a SQL warehouse ID."
  echo "      Open SQL → SQL Warehouses → Connection details → HTTP path:"
  echo "      /sql/1.0/warehouses/<THIS_PART_IS_THE_WAREHOUSE_ID>"
  echo ""
fi

if [[ "${DATABRICKS_HOST}" == *"adb-XXXXXXXXXX"* ]] || [[ "${DATABRICKS_HOST}" == *"your-workspace"* ]]; then
  echo "ERROR: DATABRICKS_HOST still has placeholder value. Use your real workspace URL, e.g.:"
  echo "  https://dbc-55dacb9d-63ec.cloud.databricks.com"
  exit 1
fi

dbx_curl() {
  local resp http_code
  resp=$(mktemp)
  http_code=$(curl -sS -w "%{http_code}" -o "$resp" "$@" || echo "curl_fail")
  if [[ "$http_code" == "curl_fail" ]] || [[ "$http_code" == "000" ]]; then
    echo "ERROR: Could not connect to ${DATABRICKS_HOST}" >&2
    echo "  - Check DATABRICKS_HOST matches your browser URL exactly" >&2
    echo "  - curl SSL error 60 = hostname does not match certificate" >&2
    cat "$resp" >&2 2>/dev/null || true
    rm -f "$resp"
    return 1
  fi
  if [[ "$http_code" -ge 400 ]]; then
    echo "ERROR: HTTP ${http_code}" >&2
    cat "$resp" >&2
    rm -f "$resp"
    return 1
  fi
  cat "$resp"
  rm -f "$resp"
}

echo "==> 1/3 SQL Warehouse (host: ${DATABRICKS_HOST})"
if ! SQL_RESP=$(dbx_curl -X POST "${DATABRICKS_HOST}/api/2.0/sql/statements" \
  -H "Authorization: Bearer ${DATABRICKS_TOKEN}" \
  -H "Content-Type: application/json" \
  -d "{\"warehouse_id\":\"${DATABRICKS_SQL_WAREHOUSE_ID}\",\"statement\":\"SELECT 1 AS ok\",\"wait_timeout\":\"30s\"}"); then
  echo "" >&2
  echo "Common fixes:" >&2
  echo "  - Start the SQL warehouse in Databricks UI (SQL → SQL Warehouses → Start)" >&2
  echo "  - Regenerate token with SQL scope: Settings → Developer → Access tokens" >&2
  echo "  - Confirm warehouse ID from Connection details HTTP path" >&2
  exit 1
fi
echo "    SQL API OK: $(echo "$SQL_RESP" | python3 -c "import sys,json; print(json.load(sys.stdin).get('status',{}).get('state','unknown'))")"

echo "==> 2/3 Gold predictions table"
if TABLE_RESP=$(dbx_curl -X POST "${DATABRICKS_HOST}/api/2.0/sql/statements" \
  -H "Authorization: Bearer ${DATABRICKS_TOKEN}" \
  -H "Content-Type: application/json" \
  -d "{\"warehouse_id\":\"${DATABRICKS_SQL_WAREHOUSE_ID}\",\"statement\":\"DESCRIBE TABLE ${CATALOG}.gold.predictions\",\"wait_timeout\":\"30s\"}" 2>/dev/null); then
  STATE=$(echo "$TABLE_RESP" | python3 -c "import sys,json; print(json.load(sys.stdin).get('status',{}).get('state','FAILED'))")
  if [[ "$STATE" == "SUCCEEDED" ]]; then
    echo "    Table ${CATALOG}.gold.predictions exists"
  else
    echo "    WARN: DESCRIBE returned state=${STATE} — run: make databricks-init-catalog"
  fi
else
  echo "    WARN: ${CATALOG}.gold.predictions not found — run: make databricks-init-catalog"
fi

echo "==> 3/3 Model Serving endpoint: ${ENDPOINT}"
SERVE_RESP=$(dbx_curl -X POST "${DATABRICKS_HOST}/serving-endpoints/${ENDPOINT}/invocations" \
  -H "Authorization: Bearer ${DATABRICKS_TOKEN}" \
  -H "Content-Type: application/json" \
  -d '{
    "dataframe_records": [{
      "surface_area": 120,
      "number_of_rooms": 5,
      "number_of_bedrooms": 3,
      "build_year": 1985,
      "energy_label": "B",
      "property_type": "terraced_house",
      "garden": true,
      "region": "Utrecht",
      "latitude": 52.0907,
      "longitude": 5.1214,
      "prediction_date": "2026-07-14"
    }]
  }' 2>/dev/null || echo '{"error":"serving_failed"}')

if echo "$SERVE_RESP" | grep -qE 'predictions|predicted_price|error'; then
  echo "    Serving response: $(echo "$SERVE_RESP" | head -c 120)..."
else
  echo "    WARN: Serving endpoint not ready or not found — create endpoint '${ENDPOINT}' in Databricks UI"
fi

echo ""
echo "Done. Set USE_MOCK_DATABRICKS=false in .env and run: make dev-full"
