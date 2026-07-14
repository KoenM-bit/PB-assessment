#!/usr/bin/env bash
# Enable Netlify deploy previews for pull requests (free tier).
set -euo pipefail

SITE_ID="${NETLIFY_SITE_ID:-2ac2e3bf-5831-4fba-8ee3-6fc63b12583d}"

if ! command -v netlify >/dev/null 2>&1; then
  echo "ERROR: Install Netlify CLI and run: netlify login" >&2
  exit 1
fi

echo "==> Enabling deploy previews for site $SITE_ID"
netlify api updateSite --data "{
  \"site_id\": \"${SITE_ID}\",
  \"body\": {
    \"build_settings\": {
      \"skip_prs\": false
    }
  }
}" >/dev/null

echo "OK: Pull request deploy previews enabled."
echo "PR URLs will look like: deploy-preview-123--pb-assessment.netlify.app"
echo "Uses netlify.toml [context.deploy-preview] env (staging/challenger)."
