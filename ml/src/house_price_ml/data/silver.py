"""Bronze to Silver transformation."""

from __future__ import annotations

import pandas as pd

from house_price_ml.data.validation import validate_listing


def bronze_to_silver(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Transform raw bronze listings to clean silver and rejected tables."""
    clean_rows: list[dict] = []
    rejected_rows: list[dict] = []

    for _, row in df.iterrows():
        result = validate_listing(row.to_dict())
        record = {**result.cleaned, "dq_flags": result.dq_flags, "is_valid": result.is_valid}
        if result.is_valid:
            clean_rows.append(record)
        else:
            rejected_rows.append(record)

    clean_df = pd.DataFrame(clean_rows) if clean_rows else pd.DataFrame()
    rejected_df = pd.DataFrame(rejected_rows) if rejected_rows else pd.DataFrame()

    if not clean_df.empty:
        clean_df = clean_df.drop_duplicates(subset=["listing_id"], keep="first")
        clean_df["is_duplicate"] = False

    return clean_df, rejected_df
