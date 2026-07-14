#!/usr/bin/env bash
# Link this GitHub repo to Databricks Repos (Git folders) for UI editing with sync.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"

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

export DATABRICKS_HOST="${DATABRICKS_HOST%/}"

REPO_URL="${DATABRICKS_REPO_URL:-}"
REPO_PATH="${DATABRICKS_REPO_PATH:-}"

if [ -z "$REPO_URL" ]; then
  REMOTE="$(git -C "$ROOT" remote get-url origin 2>/dev/null || true)"
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
  if [ -z "$REPO_URL" ]; then
    echo "Set DATABRICKS_REPO_URL to your GitHub HTTPS URL, e.g.:" >&2
    echo "  https://github.com/KoenM-bit/PB-assessment.git" >&2
    exit 1
  fi
  echo "Using git remote: $REPO_URL"
fi

if ! command -v databricks >/dev/null 2>&1; then
  echo "ERROR: Install Databricks CLI: brew install databricks/tap/databricks" >&2
  exit 1
fi

if [ -z "$REPO_PATH" ]; then
  USER_EMAIL="$(databricks current-user me -o json | python3 -c "import sys,json; print(json.load(sys.stdin)['userName'])")"
  REPO_PATH="/Repos/${USER_EMAIL}/PB-assessment"
fi

echo "==> Create Databricks Git folder at ${REPO_PATH}"
echo "    (Requires GitHub access — link GitHub in Databricks UI if prompted)"
echo ""

BRANCH="$(git -C "$ROOT" branch --show-current)"

set +e
CREATE_OUT="$(databricks repos create "$REPO_URL" gitHub --path "$REPO_PATH" 2>&1)"
CREATE_RC=$?
set -e

if [ "$CREATE_RC" -eq 0 ]; then
  echo "$CREATE_OUT"
else
  echo "$CREATE_OUT"
  if echo "$CREATE_OUT" | grep -qiE "already exists|RESOURCE_ALREADY_EXISTS"; then
    echo ""
    echo "Repo already exists — updating to branch ${BRANCH}"
    databricks repos update "$REPO_PATH" --branch "$BRANCH"
  else
    echo "" >&2
    echo "ERROR: Could not create repo at ${REPO_PATH}" >&2
    echo "Try manually: Databricks UI → Repos → Add Repo" >&2
    echo "  URL:  ${REPO_URL}" >&2
    echo "  Path: ${REPO_PATH}" >&2
    exit 1
  fi
fi

echo ""
echo "OK: Databricks repo ready at ${REPO_PATH}"
echo ""
echo "Workflow:"
echo "  • Edit notebooks/code in Databricks Repos OR locally"
echo "  • Commit + push to GitHub from either side"
echo "  • GitHub Actions bundle-deploy syncs jobs after push"
echo ""
echo "In Databricks UI:"
echo "  Repos → ${REPO_PATH} → open databricks/notebooks/"
