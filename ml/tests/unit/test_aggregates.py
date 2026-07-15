"""Tests for gold aggregate features and leakage guards."""

from __future__ import annotations

import pandas as pd

from house_price_ml.features.aggregates import compute_region_median_price_per_sqm


def _listing(
    listing_id: str,
    sale_date: str,
    sale_price: float,
    surface_area: float = 100.0,
) -> dict:
    return {
        "listing_id": listing_id,
        "sale_date": sale_date,
        "sale_price": sale_price,
        "surface_area": surface_area,
        "number_of_rooms": 4,
        "number_of_bedrooms": 2,
        "build_year": 1990,
        "energy_label": "B",
        "property_type": "terraced_house",
        "garden": True,
        "region": "Utrecht",
        "latitude": 52.0907,
        "longitude": 5.1214,
    }


def test_region_median_uses_only_past_sales():
    """Future sales in the same frame must not change an earlier row's median."""
    early = _listing("early", "2024-01-15", sale_price=300_000, surface_area=100)
    late = _listing("late", "2024-06-15", sale_price=600_000, surface_area=100)

    df_early_only = pd.DataFrame([early, early]).copy()
    df_early_only["feature_snapshot_date"] = pd.to_datetime(df_early_only["sale_date"]).dt.date
    median_before = compute_region_median_price_per_sqm(df_early_only).iloc[0][
        "region_median_price_per_sqm"
    ]

    df_with_future = pd.DataFrame([early, late]).copy()
    df_with_future["feature_snapshot_date"] = pd.to_datetime(df_with_future["sale_date"]).dt.date
    median_after = compute_region_median_price_per_sqm(df_with_future).iloc[0][
        "region_median_price_per_sqm"
    ]

    assert median_before == median_after


def test_region_median_excludes_own_sale():
    df = pd.DataFrame([_listing("solo", "2024-03-01", sale_price=400_000)])
    df["feature_snapshot_date"] = pd.to_datetime(df["sale_date"]).dt.date
    result = compute_region_median_price_per_sqm(df)
    # No prior sales → fallback default
    assert result.iloc[0]["region_median_price_per_sqm"] == 3000.0
