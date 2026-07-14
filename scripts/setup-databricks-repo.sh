#!/usr/bin/env bash
# Link this GitHub repo to Databricks Repos (Git folders) for UI editing with sync.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REPO_URL="${DATABRICKS_REPO_URL:-}"
REPO_PATH="${DATABRICKS_REPO_PATH:-}"

if [ -z "$REPO_PATH" ] && [ -n "${DATABRICKS_HOST:-}" ] && command -v databricks >/dev/null 2>&1; then
  USER_EMAIL="$(databricks current-user me -o json 2>/dev/null | python3 -c "import sys,json; print(json.load(sys.stdin).get('userName',''))" 2>/dev/null || true)"
  if [ -n "$USER_EMAIL" ]; then
    REPO_PATH="/Repos/${USER_EMAIL}/PB-assessment"
  fi
fi
REPO_PATH="${REPO_PATH:-/Repos/PB-assessment}"

if [ -f "$ROOT/.env" ]; then
  set -a
  # shellcheck disable=SC1091
  source "$ROOT/.env"
  set +a
fi

if [ -z "${DATABRICKS_HOST:-}" ]; then
  echo "ERROR: Set DATABRICKS_HOST in .env" >&2
  exit 1
fi

if [ -z "$REPO_URL" ]; then
  REMOTE="$(git -C "$ROOT" remote get-url origin 2>/dev/null || true)"
  # Databricks Repos requires HTTPS GitHub URL (not SSH).
  case "$REMOTE" in
    git@github.com:*)
      REPO_URL="https://github.com/${REMOTE#git@github.com:}"
      REPO_URL="${REPO_URL%.git}.git"
      ;;
    git@github-default:*)
      REPO_URL="https://github.com/${REMOTE#git@github-default:}"
      REPO_URL="${REPO_URL%.git}.git"
      ;;
    *)
      REPO_URL="$REMOTE"
      ;;
  esac
  if [ -n "$REPO_URL" ]; then
    echo "Using git remote: $REPO_URL"
  else
    echo "Set DATABRICKS_REPO_URL to your GitHub HTTPS URL, e.g.:" >&2
    echo "  https://github.com/KoenM-bit/PB-assessment.git" >&2
    exit 1
  fi
fi

if ! command -v databricks >/dev/null 2>&1; then
  echo "ERROR: Install Databricks CLI: brew install databricks/tap/databricks" >&2
  exit 1
fi

echo "==> Create Databricks Git folder at ${REPO_PATH}"
echo "    (If this fails, create the repo manually in Databricks UI → Repos → Add Repo)"
echo ""

set +e
databricks repos create "$REPO_URL" --path "$REPO_PATH" 2>&1
CREATE_RC=$?
set -e

if [ "$CREATE_RC" -ne 0 ]; then
  echo ""
  echo "Repo may already exist. Update from default branch:"
  databricks repos update "$REPO_PATH" --branch "$(git -C "$ROOT" branch --show-current)"
fi

echo ""
echo "OK: Databricks repo ready at ${REPO_PATH}"
echo ""
echo "Workflow:"
echo "  • Edit notebooks/code in Databricks Repos OR locally"
echo "  • Commit + push to GitHub from either side"
echo "  • Databricks jobs run code from the synced repo after bundle deploy"
echo "  • GitHub Actions runs the same commands as make (see docs/enterprise-workflow.md)"
echo ""
echo "In Databricks UI:"
echo "  Repos → ${REPO_PATH} → open databricks/notebooks/"
echo "  Workflows → Jobs → run [staging] Full ML Pipeline"
