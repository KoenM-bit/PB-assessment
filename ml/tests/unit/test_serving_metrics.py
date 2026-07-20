from datetime import date, datetime, timezone

import pandas as pd

from house_price_ml.monitoring.serving import build_daily_serving_metrics


def test_build_daily_serving_metrics_merges_predictions_and_events():
    predictions = pd.DataFrame(
        {
            "prediction_timestamp": [
                datetime(2026, 7, 18, 10, 0, tzinfo=timezone.utc),
                datetime(2026, 7, 18, 11, 0, tzinfo=timezone.utc),
            ],
            "serving_latency_ms": [120, 31_000],
        }
    )
    events = pd.DataFrame(
        {
            "event_timestamp": [datetime(2026, 7, 18, 12, 0, tzinfo=timezone.utc)],
            "http_status": [504],
            "is_timeout": [True],
            "latency_ms": [30_001],
        }
    )

    result = build_daily_serving_metrics(
        predictions,
        events,
        timeout_ms=30_000,
        as_of=date(2026, 7, 20),
    )

    assert len(result) == 1
    row = result.iloc[0]
    assert row["date"] == date(2026, 7, 18)
    assert row["request_count"] == 2
    assert row["error_count"] == 1
    assert row["timeout_count"] == 2  # one slow success + one timeout event
    assert row["p50_latency_ms"] == 120
    assert row["p95_latency_ms"] == 31_000


def test_build_daily_serving_metrics_empty_inputs():
    result = build_daily_serving_metrics(pd.DataFrame(), pd.DataFrame(), as_of=date(2026, 7, 20))
    assert result.empty
