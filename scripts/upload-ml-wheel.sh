#!/usr/bin/env bash
# Build the ML package wheel and upload to the Databricks workspace for notebooks/jobs.
# Uses the current user's workspace libs/ folder (/Workspace/Shared requires admin).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WHEEL_NAME="house_price_ml-0.1.0-py3-none-any.whl"

if [ -z "${DATABRICKS_HOST:-}" ]; then
  echo "ERROR: Set DATABRICKS_HOST" >&2
  exit 1
fi

export DATABRICKS_HOST="${DATABRICKS_HOST%/}"

if ! command -v databricks >/dev/null 2>&1; then
  echo "ERROR: Databricks CLI not found." >&2
  exit 1
fi

resolve_wheel_dest() {
  if [ -n "${DATABRICKS_ML_WHEEL_PATH:-}" ]; then
    echo "$DATABRICKS_ML_WHEEL_PATH"
    return
  fi
  local user
  user="$(databricks current-user me -o json | python3 -c "import sys, json; print(json.load(sys.stdin)['userName'])")"
  echo "/Workspace/Users/${user}/libs/${WHEEL_NAME}"
}

echo "==> Build ML wheel"
cd "$ROOT/ml"
rm -rf dist
pip wheel . -w dist/ --no-deps -q
WHEEL="$(ls -1 dist/house_price_ml-*.whl | head -1)"
echo "    Built: $(basename "$WHEEL")"

WHEEL_DEST="$(resolve_wheel_dest)"
DEST_DIR="$(dirname "$WHEEL_DEST")"
echo "==> Upload to ${WHEEL_DEST}"
databricks workspace mkdirs "$DEST_DIR" 2>/dev/null || true
databricks workspace import "$WHEEL_DEST" --file "$WHEEL" --format AUTO --overwrite

echo "OK: Wheel uploaded."
echo "    Notebooks install via wheel_path widget: ${WHEEL_DEST}"
