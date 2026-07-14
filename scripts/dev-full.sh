#!/usr/bin/env bash
# Local full-stack dev: Netlify Functions + Vite (no linked Netlify site required).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# Load .env for Databricks credentials (USE_MOCK_DATABRICKS, tokens, etc.)
if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

cleanup() {
  jobs -p | xargs kill 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "Starting Netlify Functions on http://localhost:9999"
netlify functions:serve --functions netlify/functions --port 9999 &

# Wait until functions server is accepting connections
for _ in $(seq 1 30); do
  if curl -sf "http://localhost:9999/.netlify/functions/monitoring" >/dev/null 2>&1; then
    break
  fi
  sleep 0.2
done

echo "Starting Vite on http://localhost:5173 (API proxied to functions)"
export VITE_API_PROXY_TARGET=http://localhost:9999
cd apps/web
exec npm run dev -- --host 127.0.0.1 --port 5173 --strictPort
