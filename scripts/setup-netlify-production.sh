#!/usr/bin/env bash
# Configure Netlify production (master branch) for real Databricks + champion model.
# Free tier: shares house-price-serving with staging (separate prod endpoint not required).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"

if ! command -v netlify >/dev/null 2>&1; then
  echo "ERROR: Install Netlify CLI and run: netlify login" >&2
  exit 1
fi

if [ -f "$ROOT/.env" ]; then
  set -a
  # shellcheck disable=SC1091
  source "$ROOT/.env"
  set +a
fi

set_var() {
  local key="$1"
  local value="$2"
  echo "==> ${key} (production context)"
  netlify env:set "$key" "$value" --context production --force
}

echo "==> Configuring Netlify production env for real Databricks"
set_var USE_MOCK_DATABRICKS false
set_var APP_ENV production
set_var MODEL_ALIAS champion
set_var DATABRICKS_CATALOG "${DATABRICKS_CATALOG:-house_price_staging}"
set_var DATABRICKS_SERVING_ENDPOINT "${DATABRICKS_SERVING_ENDPOINT:-house-price-serving}"
set_var DATABRICKS_SCHEMA "${DATABRICKS_SCHEMA:-gold}"
set_var SERVING_TIMEOUT_MS "${SERVING_TIMEOUT_MS:-30000}"
set_var SQL_MAX_WAIT_MS "${SQL_MAX_WAIT_MS:-25000}"

if [ -n "${DATABRICKS_HOST:-}" ]; then
  set_var DATABRICKS_HOST "$DATABRICKS_HOST"
fi
if [ -n "${DATABRICKS_TOKEN:-}" ]; then
  set_var DATABRICKS_TOKEN "$DATABRICKS_TOKEN"
fi
if [ -n "${DATABRICKS_SQL_WAREHOUSE_ID:-}" ]; then
  set_var DATABRICKS_SQL_WAREHOUSE_ID "$DATABRICKS_SQL_WAREHOUSE_ID"
fi
if [ -n "${DEMO_WRITE_TOKEN:-}" ]; then
  set_var DEMO_WRITE_TOKEN "$DEMO_WRITE_TOKEN"
fi

echo ""
echo "OK: Production env configured."
echo "Next: merge staging → master, then trigger a production deploy:"
echo "  git checkout master && git merge staging && git push origin master"
echo "  netlify deploy --prod"
