#!/usr/bin/env bash
# Wait until a Netlify deployment responds on /api/monitoring.
set -euo pipefail

BASE_URL="${1%/}"
MAX_ATTEMPTS="${2:-36}"

if [ -z "$BASE_URL" ]; then
  echo "Usage: $0 <base-url> [max-attempts]" >&2
  exit 1
fi

for attempt in $(seq 1 "$MAX_ATTEMPTS"); do
  if curl -fsS --max-time 20 "${BASE_URL}/api/monitoring" >/dev/null 2>&1; then
    echo "Deployment ready: ${BASE_URL} (attempt ${attempt}/${MAX_ATTEMPTS})"
    exit 0
  fi
  echo "Waiting for ${BASE_URL}... (${attempt}/${MAX_ATTEMPTS})"
  sleep 10
done

echo "Timeout waiting for ${BASE_URL}" >&2
exit 1
