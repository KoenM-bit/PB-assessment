"""Historical aggregate features (Silver to Gold, time-aware)."""

from __future__ import annotations

import pandas as pd


def compute_region_median_price_per_sqm(
    df: pd.DataFrame,
    snapshot_date_col: str = "feature_snapshot_date",
) -> pd.DataFrame:
    """
    Compute region + property_type median price per sqm using only past sales.

    Point-in-time: for each row, only rows with ``sale_date < feature_snapshot_date``
    are used. This allows ``silver_to_gold_features`` to run on the full dataset
    before train/test split without leaking future sale prices into earlier rows.

    Do not replace with a global ``groupby(...).transform("median")`` — that would
    leak target statistics across splits.
    """
    if df.empty:
        return df

    result = df.copy()
    medians: list[float] = []

    for idx, row in result.iterrows():
        snapshot = pd.to_datetime(row[snapshot_date_col])
        historical = df[
            (pd.to_datetime(df["sale_date"]) < snapshot)
            & (df["region"] == row["region"])
            & (df["property_type"] == row["property_type"])
            & (df["sale_price"].notna())
            & (df["surface_area"] > 0)
        ]
        if len(historical) >= 5:
            median_psm = (historical["sale_price"] / historical["surface_area"]).median()
        else:
            fallback = df[
                (pd.to_datetime(df["sale_date"]) < snapshot)
                & (df["region"] == row["region"])
                & (df["sale_price"].notna())
                & (df["surface_area"] > 0)
            ]
            median_psm = (
                (fallback["sale_price"] / fallback["surface_area"]).median()
                if len(fallback) >= 3
                else 3000.0
            )
        medians.append(float(median_psm))

    result["region_median_price_per_sqm"] = medians
    return result


def silver_to_gold_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build gold listing features from silver clean data.

    Row-wise features (house_age, surface_per_room, geo, calendar) are safe to
    compute before train/test split. Target-derived ``region_median_price_per_sqm``
    is also safe here because it is point-in-time (see
    ``compute_region_median_price_per_sqm``).

    Model training in ``train.py`` uses a separate static region-median lookup
    (fitted on train only) for serving compatibility — not the gold column directly.
    """
    from house_price_ml.features.energy import energy_label_to_score
    from house_price_ml.features.geo import distance_to_city_centre

    result = df.copy()
    result["feature_snapshot_date"] = pd.to_datetime(
        result.get("sale_date", result.get("listing_timestamp"))
    ).dt.date
    result["house_age"] = result["feature_snapshot_date"].apply(
        lambda d: d.year if hasattr(d, "year") else pd.Timestamp(d).year
    ) - result["build_year"]
    result["surface_per_room"] = result["surface_area"] / result["number_of_rooms"]
    result["energy_label_score"] = result["energy_label"].map(energy_label_to_score)
    result["surface_x_energy"] = result["surface_area"] * result["energy_label_score"]
    result["dist_to_city_centre_km"] = result.apply(
        lambda r: distance_to_city_centre(r["region"], r["latitude"], r["longitude"]),
        axis=1,
    )
    result["month"] = pd.to_datetime(result["feature_snapshot_date"]).dt.month
    result["quarter"] = pd.to_datetime(result["feature_snapshot_date"]).dt.quarter
    result["label_sale_price"] = result.get("sale_price")
    result = compute_region_median_price_per_sqm(result)
    return result
