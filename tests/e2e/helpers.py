"""HTTP helpers for live deployment smoke tests."""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Any

DEFAULT_TIMEOUT_S = 30
PREDICT_TIMEOUT_S = 90
MAX_ATTEMPTS = 3


def base_url() -> str:
    url = os.environ.get("E2E_BASE_URL") or os.environ.get("STAGING_URL")
    if not url:
        raise RuntimeError("Set E2E_BASE_URL or STAGING_URL to run live E2E tests")
    return url.rstrip("/")


def api_json(
    path: str,
    *,
    method: str = "GET",
    payload: dict[str, Any] | None = None,
    timeout_s: int = DEFAULT_TIMEOUT_S,
) -> dict[str, Any]:
    data = None
    headers = {"Content-Type": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode()

    request = urllib.request.Request(
        f"{base_url()}{path}",
        data=data,
        headers=headers,
        method=method,
    )
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        body = json.loads(response.read())
    if body.get("error"):
        raise AssertionError(body["error"].get("message", "API error"))
    return body


def api_json_with_retry(
    path: str,
    *,
    method: str = "GET",
    payload: dict[str, Any] | None = None,
    timeout_s: int = DEFAULT_TIMEOUT_S,
    attempts: int = MAX_ATTEMPTS,
) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return api_json(path, method=method, payload=payload, timeout_s=timeout_s)
        except (urllib.error.URLError, TimeoutError, AssertionError) as err:
            last_error = err
            if attempt == attempts:
                raise
            time.sleep(attempt * 5)
    raise last_error or RuntimeError("request failed")


SAMPLE_PREDICT_PAYLOAD = {
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
