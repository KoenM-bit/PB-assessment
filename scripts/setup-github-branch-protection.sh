#!/usr/bin/env bash
# Configure branch protection on GitHub (requires admin access to the repo).
#
# Prerequisites:
#   brew install gh && gh auth login
#
# Usage:
#   ./scripts/setup-github-branch-protection.sh
#
set -euo pipefail

REPO="${GITHUB_REPOSITORY:-KoenM-bit/PB-assessment}"
CHECK_NAME="required-checks"

protect_branch() {
  local branch="$1"
  echo "==> Protecting branch: $branch"

  if ! command -v gh >/dev/null 2>&1; then
    echo "ERROR: Install GitHub CLI: brew install gh && gh auth login" >&2
    exit 1
  fi

  gh api \
    --method PUT \
    "repos/${REPO}/branches/${branch}/protection" \
    --input - <<EOF
{
  "required_status_checks": {
    "strict": true,
    "checks": [
      {"context": "required-checks", "app_id": -1}
    ]
  },
  "required_pull_request_reviews": {
    "dismiss_stale_reviews": true,
    "require_code_owner_reviews": false,
    "required_approving_review_count": 1
  },
  "enforce_admins": false,
  "restrictions": null,
  "allow_force_pushes": false,
  "allow_deletions": false,
  "required_conversation_resolution": true
}
EOF

  echo "    OK: $branch requires PR + 1 approval + green required-checks"
}

echo "Repository: $REPO"
echo "Required status check: required-checks"
echo "Run CI once on a PR before applying protection if GitHub cannot find the check yet."
echo ""

protect_branch "master"
protect_branch "staging"

cat <<EOF

Done.

Typical flow:
  feature branch → PR to staging → CI + Netlify deploy preview → 1 approval → merge
  staging → PR to master → CI + deploy preview → 1 approval → merge → production deploy

EOF
