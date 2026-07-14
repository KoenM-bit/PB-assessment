#!/usr/bin/env bash
# Netlify Dev must exec the Vite process directly (not a short-lived shell/npm parent).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT/apps/web"
exec ./node_modules/.bin/vite --host 127.0.0.1 --port 5173
