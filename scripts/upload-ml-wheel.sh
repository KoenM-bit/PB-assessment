#!/usr/bin/env bash
# Build the ML package wheel and upload to the Databricks workspace for notebooks/jobs.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WHEEL_DEST="${DATABRICKS_ML_WHEEL_PATH:-/Workspace/Shared/house_price_ml-0.1.0-py3-none-any.whl}"

if [ -z "${DATABRICKS_HOST:-}" ]; then
  echo "ERROR: Set DATABRICKS_HOST" >&2
  exit 1
fi

export DATABRICKS_HOST="${DATABRICKS_HOST%/}"

if ! command -v databricks >/dev/null 2>&1; then
  echo "ERROR: Databricks CLI not found." >&2
  exit 1
fi

echo "==> Build ML wheel"
cd "$ROOT/ml"
rm -rf dist
pip wheel . -w dist/ --no-deps -q
WHEEL="$(ls -1 dist/house_price_ml-*.whl | head -1)"
echo "    Built: $(basename "$WHEEL")"

echo "==> Upload to dbfs:/Workspace/Shared/$(basename "$WHEEL")"
databricks fs cp "$WHEEL" "dbfs:/Workspace/Shared/$(basename "$WHEEL")" --overwrite

echo "OK: Wheel uploaded. Notebooks install: %pip install /Workspace/Shared/$(basename "$WHEEL")"
