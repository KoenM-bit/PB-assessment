#!/usr/bin/env bash
# Netlify build hook — installs deps and builds the React app.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "==> Installing web dependencies"
cd "$ROOT/apps/web"
npm ci
npm run build

echo "==> Installing function dependencies"
cd "$ROOT/netlify/functions"
npm ci

if [ ! -f "$ROOT/netlify/functions/_shared/training_manifest.json" ]; then
  echo "WARNING: training_manifest.json missing — run 'make train' before deploying for full monitoring data"
fi

echo "==> Netlify build complete"
