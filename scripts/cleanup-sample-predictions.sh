#!/usr/bin/env bash
# Remove sample / demo rows from gold.predictions (and linked actual_sales).
#
# Usage (from repo root, with .env configured):
#   ./scripts/cleanup-sample-predictions.sh
#   DRY_RUN=1 ./scripts/cleanup-sample-predictions.sh   # preview only
#
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

CATALOG="${DATABRICKS_CATALOG:-house_price_staging}"
DATABRICKS_HOST="${DATABRICKS_HOST%/}"
DRY_RUN="${DRY_RUN:-0}"

# Match Domstraat 12 sample listings and the 500 m² demo row.
WHERE_CLAUSE="(
  LOWER(address) LIKE '%domstraat 12%'
  OR LOWER(property_key) LIKE '%domstraat 12%'
  OR CAST(get_json_object(request_payload, '$.surface_area') AS DOUBLE) = 500
)"

run_sql() {
  local statement="$1"
  local payload
  payload=$(STATEMENT="$statement" WAREHOUSE_ID="${DATABRICKS_SQL_WAREHOUSE_ID}" python3 <<'PY'
import json, os
print(json.dumps({
    "warehouse_id": os.environ["WAREHOUSE_ID"],
    "statement": os.environ["STATEMENT"],
    "wait_timeout": "50s",
}))
PY
)
  local resp
  resp=$(curl -sS -X POST "${DATABRICKS_HOST}/api/2.0/sql/statements" \
    -H "Authorization: Bearer ${DATABRICKS_TOKEN}" \
    -H "Content-Type: application/json" \
    -d "$payload")
  local state
  state=$(echo "$resp" | python3 -c "import sys,json; print(json.load(sys.stdin).get('status',{}).get('state','FAILED'))")
  if [[ "$state" == "FAILED" ]]; then
    echo "$resp" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('status',{}).get('error',{}).get('message','SQL failed'))" >&2
    exit 1
  fi
  echo "$resp"
}

echo "Catalog: ${CATALOG}"
echo "==> Rows to remove (preview)"
run_sql "SELECT prediction_id, address, property_key,
  CAST(get_json_object(request_payload, '$.surface_area') AS DOUBLE) AS surface_area,
  prediction_timestamp
  FROM ${CATALOG}.gold.predictions
  WHERE ${WHERE_CLAUSE}
  ORDER BY prediction_timestamp DESC" | python3 -c "
import sys, json
r = json.load(sys.stdin)
cols = [c['name'] for c in r.get('manifest',{}).get('schema',{}).get('columns',[])]
rows = r.get('result',{}).get('data_array',[]) or []
print(f'count={len(rows)}')
for row in rows:
    print(dict(zip(cols, row)))
"

if [[ "$DRY_RUN" == "1" ]]; then
  echo "DRY_RUN=1 — no deletes executed."
  exit 0
fi

echo "==> Delete linked actual_sales"
run_sql "DELETE FROM ${CATALOG}.gold.actual_sales
  WHERE prediction_id IN (
    SELECT prediction_id FROM ${CATALOG}.gold.predictions WHERE ${WHERE_CLAUSE}
  )" >/dev/null

echo "==> Delete predictions"
run_sql "DELETE FROM ${CATALOG}.gold.predictions WHERE ${WHERE_CLAUSE}" >/dev/null

echo "Done."
