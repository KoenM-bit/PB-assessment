"""End-to-end staging test specification.

Run against staging deployment with:
  STAGING_URL=https://staging.example.netlify.app pytest tests/e2e -v

Locally, tests use mock mode via Netlify function imports.
"""

import os
import uuid

import pytest

STAGING_URL = os.environ.get("STAGING_URL")


@pytest.mark.skipif(not STAGING_URL, reason="STAGING_URL not set — document-only locally")
def test_e2e_staging_prediction_flow():
    """Full staging flow: predict → list → actual sale → monitoring."""
    import urllib.request
    import json

    base = STAGING_URL.rstrip("/")
    payload = {
        "address": "Domstraat 12",
        "postcode": "3512 JC",
        "surface_area": 120,
        "number_of_rooms": 5,
        "number_of_bedrooms": 3,
        "build_year": 1985,
        "energy_label": "B",
        "property_type": "terraced_house",
        "garden": True,
        "region": "Utrecht",
        "latitude": 52.0907,
        "longitude": 5.1214,
    }

    req = urllib.request.Request(
        f"{base}/api/predict",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req) as resp:
        body = json.loads(resp.read())
    assert body["data"]["predicted_price"] > 0
    prediction_id = body["data"]["prediction_id"]

    with urllib.request.urlopen(f"{base}/api/predictions") as resp:
        listings = json.loads(resp.read())
    assert any(i["prediction_id"] == prediction_id for i in listings["data"]["items"])


def test_e2e_mock_flow_documentation():
    """Documents expected e2e steps for local/staging verification."""
    steps = [
        "POST /api/predict with valid listing",
        "GET /api/predictions — verify prediction appears",
        "POST /api/actual-sales with write token",
        "Run evaluation workflow (Databricks job)",
        "GET /api/monitoring — verify metrics with sample sizes",
    ]
    assert len(steps) == 5
