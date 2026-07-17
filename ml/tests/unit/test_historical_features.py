"""Tests for point-in-time historical market features."""

from __future__ import annotations

import pandas as pd

from house_price_ml.features.historical import add_historical_market_features


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "region": ["Amsterdam", "Amsterdam", "Amsterdam"],
            "property_type": ["apartment", "apartment", "apartment"],
            "surface_area": [80.0, 90.0, 100.0],
            "label_sale_price": [320000.0, 360000.0, 400000.0],
            "sale_date": pd.to_datetime(
                ["2023-01-15", "2023-06-15", "2023-12-15"]
            ),
        }
    )


def test_historical_features_use_only_past_sales():
    df = _frame()
    enriched = add_historical_market_features(df, min_history_count=1)

    first = enriched.iloc[0]
    last = enriched.iloc[-1]

    assert first["historic_region_sale_count"] == 0
    assert last["historic_region_sale_count"] == 2
    assert last["historic_region_property_sale_count"] == 2
    assert last["historic_region_median_price_per_sqm"] > 0


def test_historical_value_estimate_scales_with_surface():
    df = _frame()
    enriched = add_historical_market_features(df, min_history_count=1)
    row = enriched.iloc[-1]

    assert row["historic_region_value_estimate"] == (
        row["historic_region_median_price_per_sqm"] * row["surface_area"]
    )
