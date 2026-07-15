"""Shared sklearn feature pipeline for training and serving."""

from __future__ import annotations

from datetime import date
from typing import Any

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from house_price_ml.config.constants import PROPERTY_TYPES, REGIONS
from house_price_ml.features.energy import energy_label_to_score
from house_price_ml.features.geo import distance_to_city_centre

NUMERIC_FEATURES = [
    "surface_area",
    "number_of_rooms",
    "number_of_bedrooms",
    "house_age",
    "surface_per_room",
    "energy_label_score",
    "surface_x_energy",
    "dist_to_city_centre_km",
    "region_median_price_per_sqm",
    "month",
    "quarter",
    "garden_numeric",
]

CATEGORICAL_FEATURES = ["region", "property_type", "energy_label"]

FEATURE_GROUPS: dict[str, list[str]] = {
    "geo": ["dist_to_city_centre_km"],
    "energy": ["energy_label_score", "surface_x_energy", "energy_label"],
    "calendar": ["month", "quarter"],
    "region_median": ["region_median_price_per_sqm"],
    "interactions": ["surface_x_energy"],
}

def compute_row_features(row: dict[str, Any], region_medians: dict[tuple[str, str], float] | None = None) -> dict[str, Any]:
    """Compute model-ready row features from raw listing input."""
    prediction_date = row.get("prediction_date") or row.get("feature_snapshot_date") or date.today()
    if isinstance(prediction_date, str):
        prediction_date = date.fromisoformat(prediction_date[:10])
    elif hasattr(prediction_date, "date"):
        prediction_date = prediction_date.date()

    region = row["region"]
    prop_type = row["property_type"]
    median_key = (region, prop_type)
    default_median = 3200.0
    region_median = (region_medians or {}).get(median_key, default_median)

    house_age = prediction_date.year - int(row["build_year"])
    surface = float(row["surface_area"])
    rooms = int(row["number_of_rooms"])
    energy_score = energy_label_to_score(str(row["energy_label"]))

    return {
        "surface_area": surface,
        "number_of_rooms": rooms,
        "number_of_bedrooms": int(row["number_of_bedrooms"]),
        "house_age": house_age,
        "surface_per_room": surface / rooms,
        "energy_label_score": energy_score,
        "surface_x_energy": surface * energy_score,
        "dist_to_city_centre_km": distance_to_city_centre(
            region, float(row["latitude"]), float(row["longitude"])
        ),
        "region_median_price_per_sqm": region_median,
        "month": prediction_date.month,
        "quarter": (prediction_date.month - 1) // 3 + 1,
        "garden_numeric": 1.0 if row.get("garden") else 0.0,
        "region": region,
        "property_type": prop_type,
        "energy_label": str(row["energy_label"]),
    }


def raw_to_feature_frame(
    rows: list[dict[str, Any]],
    region_medians: dict[tuple[str, str], float] | None = None,
) -> pd.DataFrame:
    """Convert raw listing dicts to feature DataFrame."""
    features = [compute_row_features(r, region_medians) for r in rows]
    return pd.DataFrame(features)


def build_preprocessor() -> ColumnTransformer:
    """Build sklearn preprocessor (no estimator)."""
    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(
                    categories=[REGIONS, PROPERTY_TYPES, ["A++", "A+", "A", "B", "C", "D", "E", "F", "G"]],
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )
    return ColumnTransformer(
        [
            ("num", numeric_pipeline, NUMERIC_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )


def get_training_feature_bounds(
    df: pd.DataFrame,
    region_medians: dict[tuple[str, str], float] | None = None,
) -> dict[str, dict[str, float]]:
    """Compute p01/p99 bounds for out-of-range warnings at inference."""
    bounds: dict[str, dict[str, float]] = {}
    feature_df = raw_to_feature_frame(df.to_dict("records"), region_medians)
    for col in NUMERIC_FEATURES:
        if col in feature_df.columns:
            p01 = float(feature_df[col].quantile(0.01))
            p99 = float(feature_df[col].quantile(0.99))
            if col == "dist_to_city_centre_km":
                p01 = 0.0  # distance is non-negative; city-centre listings are valid
            bounds[col] = {"p01": p01, "p99": p99}
    return bounds


def check_out_of_range(features: dict[str, Any], bounds: dict[str, dict[str, float]]) -> list[str]:
    """Return warnings for features outside training distribution."""
    warnings: list[str] = []
    for col, lim in bounds.items():
        val = features.get(col)
        if val is None:
            continue
        # Skip degenerate bounds (e.g. when bounds were computed without region medians).
        if lim["p99"] - lim["p01"] < 1e-6:
            continue
        if val < lim["p01"] or val > lim["p99"]:
            warnings.append(f"{col} outside training range ({lim['p01']:.1f} - {lim['p99']:.1f})")
    return warnings
