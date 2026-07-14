#!/usr/bin/env bash
# Apply medallion DDL to a Unity Catalog (substitutes ${catalog} in SQL files).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec python3 "$ROOT/scripts/databricks-init-catalog.py" "${1:-house_price_staging}"
