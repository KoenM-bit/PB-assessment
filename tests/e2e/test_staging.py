"""End-to-end smoke tests against a live Netlify deployment.

Run locally:
  E2E_BASE_URL=https://staging--pb-assessment.netlify.app pytest tests/e2e -v
"""

import os

import pytest

from .helpers import (
    PREDICT_TIMEOUT_S,
    SAMPLE_PREDICT_PAYLOAD,
    api_json_with_retry,
    base_url,
)

LIVE_URL = os.environ.get("E2E_BASE_URL") or os.environ.get("STAGING_URL")


@pytest.mark.skipif(not LIVE_URL, reason="E2E_BASE_URL / STAGING_URL not set")
def test_e2e_monitoring_endpoint():
    body = api_json_with_retry("/api/monitoring")
    summary = body["data"]["summary"]
    assert summary["total_predictions"] >= 0
    assert summary["active_model_version"]


@pytest.mark.skipif(not LIVE_URL, reason="E2E_BASE_URL / STAGING_URL not set")
def test_e2e_prediction_flow():
    """predict → list predictions (full API path used by the UI)."""
    predict = api_json_with_retry(
        "/api/predict",
        method="POST",
        payload=SAMPLE_PREDICT_PAYLOAD,
        timeout_s=PREDICT_TIMEOUT_S,
        attempts=3,
    )
    assert predict["data"]["predicted_price"] > 0
    prediction_id = predict["data"]["prediction_id"]

    listings = api_json_with_retry("/api/predictions")
    items = listings["data"]["items"]
    assert any(item["prediction_id"] == prediction_id for item in items)


def test_e2e_flow_documentation():
    """Documents expected manual verification steps beyond automated smoke."""
    steps = [
        "POST /api/predict with valid listing",
        "GET /api/predictions — verify prediction appears",
        "POST /api/actual-sales with write token",
        "Run evaluation workflow (Databricks job)",
        "GET /api/monitoring — verify metrics with sample sizes",
    ]
    assert len(steps) == 5
    if LIVE_URL:
        assert base_url().startswith("http")
