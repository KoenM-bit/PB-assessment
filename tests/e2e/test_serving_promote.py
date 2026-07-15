"""Strict live inference checks after promote + deploy serving.

Run via:
  E2E_REQUIRE_LIVE_SERVING=true pytest tests/e2e/test_serving_promote.py -v
"""

import os

import pytest

from .helpers import PREDICT_TIMEOUT_S, SAMPLE_PREDICT_PAYLOAD, api_json_with_retry

LIVE_URL = os.environ.get("E2E_BASE_URL") or os.environ.get("STAGING_URL")
REQUIRE_LIVE = os.environ.get("E2E_REQUIRE_LIVE_SERVING", "").lower() in {"1", "true", "yes"}
EXPECTED_ALIAS = os.environ.get("E2E_EXPECTED_MODEL_ALIAS", "challenger")
FALLBACK_VERSIONS = {"baseline", "mock-v1", "unknown"}


@pytest.mark.skipif(not LIVE_URL or not REQUIRE_LIVE, reason="promote verification not requested")
def test_live_model_serving_via_netlify_api():
    """Full app path must use real Databricks serving, not baseline fallback."""
    predict = api_json_with_retry(
        "/api/predict",
        method="POST",
        payload=SAMPLE_PREDICT_PAYLOAD,
        timeout_s=PREDICT_TIMEOUT_S,
        attempts=3,
    )
    data = predict["data"]
    assert data["predicted_price"] > 0
    assert data["model_version"] not in FALLBACK_VERSIONS
    assert data["model_alias"] == EXPECTED_ALIAS
    assert "fallback_to_business_baseline" not in (data.get("warnings") or [])
