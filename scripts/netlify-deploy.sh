#!/usr/bin/env bash
# Deploy to Netlify.
#
# Prerequisites:
#   1. netlify CLI installed and logged in:  netlify login
#   2. Site linked (once):                  netlify link
#   3. Env vars set in Netlify UI (see netlify.env.example)
#   4. Databricks migration applied:        make databricks-init-catalog
#
# Usage:
#   ./scripts/netlify-deploy.sh                 # draft deploy (preview URL)
#   ./scripts/netlify-deploy.sh --prod          # production deploy on linked site
#   ./scripts/netlify-deploy.sh --prod staging  # deploy with staging context
#
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if ! command -v netlify >/dev/null 2>&1; then
  echo "ERROR: Install Netlify CLI: npm install -g netlify-cli" >&2
  exit 1
fi

PROD_FLAG=""
CONTEXT_ARGS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --prod)
      PROD_FLAG="--prod"
      shift
      ;;
    --context)
      CONTEXT_ARGS+=(--context "$2")
      shift 2
      ;;
    staging|production)
      CONTEXT_ARGS+=(--context "$1")
      shift
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

echo "==> Building and deploying to Netlify"
if [ -n "$PROD_FLAG" ]; then
  netlify deploy --build --prod "${CONTEXT_ARGS[@]}"
else
  netlify deploy --build "${CONTEXT_ARGS[@]}"
fi

echo ""
echo "Done. Verify with:"
echo "  curl -s https://YOUR-SITE.netlify.app/api/monitoring"
echo "  Open the site → New Prediction → Predictions & Sales"
