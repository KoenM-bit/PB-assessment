"""Daily serving metrics rollups for gold.serving_metrics."""

from __future__ import annotations

import math
from datetime import date, timedelta

import pandas as pd


def _to_date_series(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, utc=True, errors="coerce").dt.date


def _percentile(sorted_values: list[float], p: float) -> float:
    if not sorted_values:
        return 0.0
    rank = math.ceil((p / 100) * len(sorted_values)) - 1
    return sorted_values[max(0, min(rank, len(sorted_values) - 1))]


def _latency_percentiles(latencies: pd.Series) -> tuple[float, float]:
    if latencies.empty:
        return 0.0, 0.0
    clean = sorted(latencies.dropna().astype(float).tolist())
    if not clean:
        return 0.0, 0.0
    return _percentile(clean, 50), _percentile(clean, 95)


def build_daily_serving_metrics(
    predictions: pd.DataFrame,
    events: pd.DataFrame | None = None,
    *,
    timeout_ms: int = 30_000,
    lookback_days: int = 90,
    as_of: date | None = None,
) -> pd.DataFrame:
    """Aggregate predictions and API error events into daily serving_metrics rows."""
    today = as_of or date.today()
    min_date = today - timedelta(days=lookback_days)

    pred = predictions.copy() if predictions is not None else pd.DataFrame()
    ev = events.copy() if events is not None else pd.DataFrame()

    if not pred.empty and "prediction_timestamp" in pred.columns:
        pred["_day"] = _to_date_series(pred["prediction_timestamp"])
        pred = pred[pred["_day"].notna() & (pred["_day"] >= min_date)]
    else:
        pred = pd.DataFrame(columns=["_day", "serving_latency_ms"])

    if not ev.empty and "event_timestamp" in ev.columns:
        ev["_day"] = _to_date_series(ev["event_timestamp"])
        ev = ev[ev["_day"].notna() & (ev["_day"] >= min_date)]
    else:
        ev = pd.DataFrame(columns=["_day", "http_status", "is_timeout", "latency_ms"])

    days = sorted(set(pred.get("_day", pd.Series(dtype=object)).dropna().tolist()) | set(ev.get("_day", pd.Series(dtype=object)).dropna().tolist()))
    rows: list[dict] = []

    for day in days:
        day_pred = pred[pred["_day"] == day] if not pred.empty else pred
        day_ev = ev[ev["_day"] == day] if not ev.empty else ev

        latencies = (
            day_pred["serving_latency_ms"]
            if not day_pred.empty and "serving_latency_ms" in day_pred.columns
            else pd.Series(dtype=float)
        )
        p50, p95 = _latency_percentiles(latencies)

        slow_success_timeouts = 0
        if not latencies.empty:
            slow_success_timeouts = int((latencies.astype(float) >= timeout_ms).sum())

        error_count = 0
        timeout_count = slow_success_timeouts
        if not day_ev.empty:
            error_count = len(day_ev)
            if "is_timeout" in day_ev.columns:
                timeout_count += int(day_ev["is_timeout"].fillna(False).astype(bool).sum())
            elif "error_code" in day_ev.columns:
                timeout_count += int(day_ev["error_code"].astype(str).str.upper().eq("TIMEOUT").sum())

        rows.append(
            {
                "date": day,
                "request_count": int(len(day_pred)),
                "error_count": error_count,
                "timeout_count": timeout_count,
                "p50_latency_ms": round(p50, 2),
                "p95_latency_ms": round(p95, 2),
            }
        )

    return pd.DataFrame(rows)
