#!/usr/bin/env bash
# Run live E2E smoke tests against staging or a Netlify deploy preview.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [ -n "${E2E_BASE_URL:-}" ]; then
  BASE_URL="${E2E_BASE_URL%/}"
elif [ -n "${STAGING_URL:-}" ]; then
  BASE_URL="${STAGING_URL%/}"
else
  echo "Set E2E_BASE_URL or STAGING_URL" >&2
  exit 1
fi

chmod +x "$ROOT/scripts/wait-for-deploy.sh"
"$ROOT/scripts/wait-for-deploy.sh" "$BASE_URL"

export E2E_BASE_URL="$BASE_URL"
pip install -q pytest
pytest tests/e2e/test_staging.py -v
