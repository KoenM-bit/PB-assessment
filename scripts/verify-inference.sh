#!/usr/bin/env bash
# Verify Databricks serving + (staging) Netlify inference after promote/deploy.
# Usage: ./scripts/verify-inference.sh [staging|production] [--skip-e2e]
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PROFILE="${1:-staging}"
shift || true

cd "$ROOT/ml"
if ! python -c "import mlflow" 2>/dev/null; then
  pip install -q -e ".[dev]"
fi

exec python "$ROOT/scripts/verify-inference.py" --profile "$PROFILE" "$@"
