#!/usr/bin/env bash
# Deploy Databricks Asset Bundle jobs (bronze/silver/gold/train/monitoring).
# Usage: ./scripts/databricks-bundle-deploy.sh [staging|prod]
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TARGET="${1:-staging}"

if [ -f "$ROOT/.env" ]; then
  set -a
  # shellcheck disable=SC1091
  source "$ROOT/.env"
  set +a
fi

if [ -z "${DATABRICKS_HOST:-}" ]; then
  echo "ERROR: Set DATABRICKS_HOST in .env or export it before deploying." >&2
  echo "  Example: databricks auth login --host https://YOUR-WORKSPACE.cloud.databricks.com" >&2
  exit 1
fi

export DATABRICKS_HOST="${DATABRICKS_HOST%/}"

if ! command -v databricks >/dev/null 2>&1; then
  echo "ERROR: Databricks CLI not found. Install: brew install databricks/tap/databricks" >&2
  exit 1
fi

echo "==> Validating bundle (target=${TARGET}, host=${DATABRICKS_HOST})"
(cd "$ROOT/databricks" && databricks bundle validate -t "$TARGET")

echo "==> Deploying bundle (target=${TARGET})"
(cd "$ROOT/databricks" && databricks bundle deploy -t "$TARGET")

echo ""
echo "OK. Jobs are in Databricks → Workflows → Jobs & Pipelines."
echo "Upload the ML wheel before running notebooks:"
echo "  cd ml && pip wheel . -w dist/"
echo "  databricks fs cp dist/house_price_ml-*.whl dbfs:/Workspace/Shared/house_price_ml-0.1.0-py3-none-any.whl"
