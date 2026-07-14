#!/usr/bin/env bash
# Set shared Databricks secrets in Netlify (same values for all deploy contexts).
# Staging vs production routing is handled by netlify.toml + config.ts (free tier).
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
  echo "==> ${key} (all contexts)"
  netlify env:set "$key" "$value" --force
}

echo "==> Configuring shared Netlify env (secrets only; routing is in netlify.toml)"
set_var USE_MOCK_DATABRICKS false
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
echo "OK: Shared secrets configured."
echo "Staging/production endpoint routing is automatic via APP_ENV (see netlify.toml)."
echo "Redeploy both branches after merging:"
echo "  git push origin staging && git push origin master"
