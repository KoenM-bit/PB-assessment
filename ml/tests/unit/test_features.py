"""Unit tests for core ML functions."""

from datetime import date

import numpy as np
import pytest

from house_price_ml.data.validation import validate_listing
from house_price_ml.features.energy import energy_label_to_score
from house_price_ml.features.geo import distance_to_city_centre, haversine_km
from house_price_ml.features.pipeline import compute_row_features
from house_price_ml.evaluation.metrics import bias, compute_metrics, mae, mape, rmse


def test_haversine_amsterdam_utrecht():
    dist = haversine_km(52.3676, 4.9041, 52.0907, 5.1214)
    assert 30 < dist < 50


def test_distance_to_city_centre():
    dist = distance_to_city_centre("Utrecht", 52.0907, 5.1214)
    assert dist < 1.0


def test_energy_label_to_score():
    assert energy_label_to_score("A++") == 10
    assert energy_label_to_score("G") == 2
    assert energy_label_to_score("unknown") == 0


def test_house_age_calculation():
    row = {
        "surface_area": 100,
        "number_of_rooms": 4,
        "number_of_bedrooms": 2,
        "build_year": 1990,
        "energy_label": "B",
        "property_type": "terraced_house",
        "garden": True,
        "region": "Utrecht",
        "latitude": 52.09,
        "longitude": 5.12,
        "prediction_date": "2026-07-14",
    }
    features = compute_row_features(row)
    assert features["house_age"] == 2026 - 1990


def test_validate_listing_rejects_negative_surface():
    result = validate_listing({"listing_id": "1", "surface_area": -10, "number_of_rooms": 3})
    assert not result.is_valid
    assert "invalid_surface_area" in result.dq_flags


def test_validate_listing_rejects_future_build_year():
    result = validate_listing(
        {
            "listing_id": "1",
            "surface_area": 100,
            "number_of_rooms": 4,
            "number_of_bedrooms": 2,
            "build_year": date.today().year + 5,
            "energy_label": "B",
            "property_type": "apartment",
            "region": "Utrecht",
            "latitude": 52.09,
            "longitude": 5.12,
        }
    )
    assert not result.is_valid


def test_metrics():
    y_true = np.array([400000, 500000, 450000])
    y_pred = np.array([410000, 490000, 460000])
    assert mae(y_true, y_pred) == pytest.approx(10000, rel=0.01)
    assert rmse(y_true, y_pred) > 0
    assert bias(y_true, y_pred) == pytest.approx(3333.33, rel=0.01)
    assert mape(y_true, y_pred) > 0
    metrics = compute_metrics(y_true, y_pred)
    assert "mae" in metrics and "rmse" in metrics
