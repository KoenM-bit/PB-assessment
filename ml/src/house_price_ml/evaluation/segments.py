"""Segment bucketing for evaluation."""

from __future__ import annotations

import pandas as pd

from house_price_ml.config.constants import PRICE_CATEGORY_BOUNDS


def price_category(price: float) -> str:
    for low, high, label in PRICE_CATEGORY_BOUNDS:
        if low <= price < high:
            return label
    return "premium"


def add_evaluation_segments(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result["price_category"] = result["actual_sale_price"].apply(price_category)
    return result
