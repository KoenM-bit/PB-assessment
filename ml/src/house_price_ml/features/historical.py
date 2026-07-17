"""Point-in-time historical market features for experiment notebooks."""

from __future__ import annotations

import pandas as pd

_DEFAULT_MEDIAN_PSM = 3000.0


def _snapshot_series(df: pd.DataFrame, date_column: str, snapshot_column: str | None) -> pd.Series:
    if snapshot_column and snapshot_column in df.columns:
        return pd.to_datetime(df[snapshot_column], errors="coerce")
    return pd.to_datetime(df[date_column], errors="coerce")


def _valid_history(df: pd.DataFrame, date_column: str) -> pd.DataFrame:
    price_col = "label_sale_price" if "label_sale_price" in df.columns else "sale_price"
    valid = df.copy()
    valid["_sale_date"] = pd.to_datetime(valid[date_column], errors="coerce")
    valid["_price"] = pd.to_numeric(valid[price_col], errors="coerce")
    valid["_surface"] = pd.to_numeric(valid["surface_area"], errors="coerce")
    valid = valid[
        valid["_sale_date"].notna()
        & valid["_price"].notna()
        & valid["_surface"].notna()
        & (valid["_surface"] > 0)
    ].copy()
    valid["_psm"] = valid["_price"] / valid["_surface"]
    return valid


def _median_psm(historical: pd.DataFrame, *, min_count: int, default: float) -> float:
    if len(historical) >= min_count:
        return float(historical["_psm"].median())
    return default


def add_historical_market_features(
    df: pd.DataFrame,
    *,
    date_column: str = "sale_date",
    snapshot_column: str | None = "feature_snapshot_date",
    rolling_window_days: int = 365,
    min_history_count: int = 5,
    default_median_psm: float = _DEFAULT_MEDIAN_PSM,
) -> pd.DataFrame:
    """
    Add point-in-time regional market aggregates for each row.

    Only sales strictly before the row snapshot date are used, so features are safe
    to compute on the full frame before train/validation/test splitting.
    """
    if df.empty:
        return df.copy()

    result = df.copy()
    history = _valid_history(result, date_column)
    snapshots = _snapshot_series(result, date_column, snapshot_column)

    region_medians: list[float] = []
    region_property_medians: list[float] = []
    region_12m_medians: list[float] = []
    region_property_12m_medians: list[float] = []
    region_counts: list[int] = []
    region_property_counts: list[int] = []
    region_12m_counts: list[int] = []
    region_property_12m_counts: list[int] = []
    region_estimates: list[float] = []
    region_property_estimates: list[float] = []
    region_12m_estimates: list[float] = []
    region_property_12m_estimates: list[float] = []

    for idx, row in result.iterrows():
        snapshot = snapshots.loc[idx]
        if pd.isna(snapshot):
            region_medians.append(default_median_psm)
            region_property_medians.append(default_median_psm)
            region_12m_medians.append(default_median_psm)
            region_property_12m_medians.append(default_median_psm)
            region_counts.append(0)
            region_property_counts.append(0)
            region_12m_counts.append(0)
            region_property_12m_counts.append(0)
            surface = float(row.get("surface_area") or 0.0)
            region_estimates.append(default_median_psm * surface)
            region_property_estimates.append(default_median_psm * surface)
            region_12m_estimates.append(default_median_psm * surface)
            region_property_12m_estimates.append(default_median_psm * surface)
            continue

        window_start = snapshot - pd.Timedelta(days=rolling_window_days)
        past = history[history["_sale_date"] < snapshot]
        past_12m = past[past["_sale_date"] >= window_start]

        region_hist = past[past["region"] == row["region"]]
        region_property_hist = region_hist[region_hist["property_type"] == row["property_type"]]
        region_hist_12m = past_12m[past_12m["region"] == row["region"]]
        region_property_hist_12m = region_hist_12m[
            region_hist_12m["property_type"] == row["property_type"]
        ]

        region_median = _median_psm(
            region_hist,
            min_count=min_history_count,
            default=default_median_psm,
        )
        region_property_median = _median_psm(
            region_property_hist,
            min_count=min_history_count,
            default=region_median,
        )
        region_12m_median = _median_psm(
            region_hist_12m,
            min_count=min_history_count,
            default=region_median,
        )
        region_property_12m_median = _median_psm(
            region_property_hist_12m,
            min_count=min_history_count,
            default=region_property_median,
        )

        surface = float(row.get("surface_area") or 0.0)
        region_medians.append(region_median)
        region_property_medians.append(region_property_median)
        region_12m_medians.append(region_12m_median)
        region_property_12m_medians.append(region_property_12m_median)
        region_counts.append(len(region_hist))
        region_property_counts.append(len(region_property_hist))
        region_12m_counts.append(len(region_hist_12m))
        region_property_12m_counts.append(len(region_property_hist_12m))
        region_estimates.append(region_median * surface)
        region_property_estimates.append(region_property_median * surface)
        region_12m_estimates.append(region_12m_median * surface)
        region_property_12m_estimates.append(region_property_12m_median * surface)

    result["historic_region_median_price_per_sqm"] = region_medians
    result["historic_region_property_median_price_per_sqm"] = region_property_medians
    result["historic_region_12m_median_price_per_sqm"] = region_12m_medians
    result["historic_region_property_12m_median_price_per_sqm"] = region_property_12m_medians
    result["historic_region_sale_count"] = region_counts
    result["historic_region_property_sale_count"] = region_property_counts
    result["historic_region_12m_sale_count"] = region_12m_counts
    result["historic_region_property_12m_sale_count"] = region_property_12m_counts
    result["historic_region_value_estimate"] = region_estimates
    result["historic_region_property_value_estimate"] = region_property_estimates
    result["historic_region_12m_value_estimate"] = region_12m_estimates
    result["historic_region_property_12m_value_estimate"] = region_property_12m_estimates
    return result
