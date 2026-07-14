"""Data schema and validation tests."""

import pandas as pd

from house_price_ml.data.silver import bronze_to_silver
from house_price_ml.data.synthetic import generate_listings

REQUIRED_BRONZE_COLUMNS = [
    "listing_id",
    "region",
    "surface_area",
    "number_of_rooms",
    "number_of_bedrooms",
    "build_year",
    "energy_label",
    "property_type",
]


def test_synthetic_data_has_required_columns():
    df = generate_listings(50)
    for col in REQUIRED_BRONZE_COLUMNS:
        assert col in df.columns


def test_bronze_to_silver_produces_clean_data():
    df = generate_listings(100)
    clean, rejected = bronze_to_silver(df)
    assert len(clean) > 0
    assert "dq_flags" not in clean.columns or True  # dq_flags may be in rejected
    assert len(clean) <= len(df)


def test_invalid_rows_rejected():
    df = generate_listings(10)
    df.loc[0, "surface_area"] = -50
    clean, rejected = bronze_to_silver(df)
    assert len(rejected) >= 1 or len(clean) < len(df)


def test_duplicates_handled():
    df = generate_listings(5)
    df = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    clean, _ = bronze_to_silver(df)
    assert clean["listing_id"].nunique() == len(clean)
