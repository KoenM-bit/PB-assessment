#!/usr/bin/env bash
# Deploy model + serving endpoint for staging or production.
# Usage: ./scripts/deploy-serving.sh [staging|production]
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROFILE="${1:-staging}"

case "$PROFILE" in
  staging)
    export DATABRICKS_CATALOG="${DATABRICKS_CATALOG:-house_price_staging}"
    export DATABRICKS_SERVING_ENDPOINT="${DATABRICKS_SERVING_ENDPOINT:-house-price-serving}"
    export MODEL_ALIAS="${MODEL_ALIAS:-challenger}"
    ;;
  production)
    export DATABRICKS_CATALOG="${DATABRICKS_CATALOG:-house_price_prod}"
    export DATABRICKS_SERVING_ENDPOINT="${DATABRICKS_SERVING_ENDPOINT:-house-price-serving-prod}"
    export MODEL_ALIAS="${MODEL_ALIAS:-champion}"
    ;;
  *)
    echo "Usage: $0 [staging|production]" >&2
    exit 1
    ;;
esac

echo "==> Deploy profile: ${PROFILE}"
echo "    catalog=${DATABRICKS_CATALOG}"
echo "    endpoint=${DATABRICKS_SERVING_ENDPOINT}"
echo "    alias=${MODEL_ALIAS}"
echo ""

if [[ ! -d "$ROOT/ml/artifacts/model/mlflow_model" ]]; then
  echo "Training model first..."
  make -C "$ROOT" train
fi

cd "$ROOT/ml"
if ! python -c "import mlflow" 2>/dev/null; then
  echo "Installing ML dependencies..."
  pip install -e ".[dev]"
fi

exec python "$ROOT/scripts/deploy-serving.py"
