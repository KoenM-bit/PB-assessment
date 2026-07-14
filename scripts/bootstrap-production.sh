#!/usr/bin/env bash
# Bootstrap production catalog + model registry + serving endpoint.
# Safe to run while staging endpoint is still deploying (uses a separate endpoint name).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "==> 1/2 Initialize production Unity Catalog tables"
make -C "$ROOT" databricks-init-catalog CATALOG=house_price_prod

echo ""
echo "==> 2/2 Deploy production model + serving endpoint"
"$ROOT/scripts/deploy-serving.sh" production
