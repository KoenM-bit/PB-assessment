#!/usr/bin/env bash
# Run Databricks operations from CI or locally (same entrypoint as GitHub Actions).
# Usage: ./scripts/databricks-ci.sh <command> [staging|prod]
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
COMMAND="${1:-}"
TARGET="${2:-staging}"

if [ -f "$ROOT/.env" ]; then
  set -a
  # shellcheck disable=SC1091
  source "$ROOT/.env"
  set +a
fi

if [ -z "${DATABRICKS_HOST:-}" ]; then
  echo "ERROR: Set DATABRICKS_HOST (GitHub secret or .env)." >&2
  exit 1
fi

export DATABRICKS_HOST="${DATABRICKS_HOST%/}"

require_token() {
  if [ -z "${DATABRICKS_TOKEN:-}" ]; then
    echo "ERROR: Set DATABRICKS_TOKEN (GitHub secret or .env)." >&2
    exit 1
  fi
}

bundle_target() {
  case "$1" in
    staging) echo "staging" ;;
    prod|production) echo "prod" ;;
    *) echo "ERROR: target must be staging or prod" >&2; exit 1 ;;
  esac
}

serving_profile() {
  case "$1" in
    staging) echo "staging" ;;
    prod|production) echo "production" ;;
    *) echo "ERROR: target must be staging or prod" >&2; exit 1 ;;
  esac
}

BTARGET="$(bundle_target "$TARGET")"
SPROFILE="$(serving_profile "$TARGET")"

case "$COMMAND" in
  bundle-deploy)
    require_token
    "$ROOT/scripts/databricks-bundle-deploy.sh" "$BTARGET"
    ;;
  upload-wheel)
    require_token
    "$ROOT/scripts/upload-ml-wheel.sh"
    ;;
  run-pipeline)
    require_token
    GIT_SHA="${GITHUB_SHA:-$(git -C "$ROOT" rev-parse HEAD 2>/dev/null || echo unknown)}"
    echo "==> Run full ML pipeline (target=${BTARGET}, git_commit=${GIT_SHA})"
    (cd "$ROOT/databricks" && databricks bundle run full_ml_pipeline -t "$BTARGET" --var "git_commit=${GIT_SHA}")
    ;;
  deploy-serving)
    require_token
    export FROM_REGISTRY=true
    "$ROOT/scripts/deploy-serving.sh" "$SPROFILE"
    ;;
  deploy-serving-local)
    require_token
    pip install -q -e "$ROOT/ml[dev]"
    make -C "$ROOT" train
    "$ROOT/scripts/deploy-serving.sh" "$SPROFILE"
    ;;
  promote-champion)
    require_token
    cd "$ROOT/ml" && python "$ROOT/scripts/promote-champion.py"
    ;;
  promote-to-production)
    require_token
    if [ "${CONFIRM_PROMOTE:-}" != "yes" ]; then
      echo "Set CONFIRM_PROMOTE=yes to promote staging @challenger to production." >&2
      exit 1
    fi
    cd "$ROOT/ml" && python "$ROOT/scripts/promote-to-production.py"
    ;;
  verify)
    require_token
    "$ROOT/scripts/verify-databricks.sh"
    ;;
  init-catalog)
    require_token
    CATALOG="house_price_staging"
    [ "$BTARGET" = "prod" ] && CATALOG="house_price_prod"
    "$ROOT/scripts/databricks-init-catalog.sh" "$CATALOG"
    ;;
  bootstrap-production)
    require_token
    "$ROOT/scripts/bootstrap-production.sh"
    ;;
  staging-pipeline)
    require_token
    "$0" bundle-deploy staging
    "$0" run-pipeline staging
    "$0" deploy-serving staging
    ;;
  production-pipeline)
    require_token
    "$0" bundle-deploy prod
    "$0" upload-wheel prod
    CONFIRM_PROMOTE=yes "$0" promote-to-production prod
    ;;
  *)
    echo "Usage: $0 <command> [staging|prod]" >&2
    echo "" >&2
    echo "Commands:" >&2
    echo "  bundle-deploy          Deploy Databricks Asset Bundle jobs" >&2
    echo "  upload-wheel           Build + upload house_price_ml wheel to workspace" >&2
    echo "  run-pipeline           Run bronze→silver→gold→train→evaluate job" >&2
    echo "  deploy-serving         Point serving endpoint at registry alias (no local train)" >&2
    echo "  deploy-serving-local   Train locally, register, deploy (legacy laptop flow)" >&2
    echo "  promote-champion       Move challenger alias to champion (same catalog)" >&2
    echo "  promote-to-production  Copy staging challenger → prod champion + deploy" >&2
    echo "  verify                 Health-check Databricks connectivity" >&2
    echo "  init-catalog           Create Unity Catalog tables" >&2
    echo "  bootstrap-production   Init prod catalog + deploy prod serving" >&2
    echo "  staging-pipeline       bundle + wheel + pipeline + deploy-serving (staging)" >&2
    echo "  production-pipeline    bundle prod + promote-to-production" >&2
    exit 1
    ;;
esac
