"""Listing validation rules for Silver layer."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any

from house_price_ml.config.constants import (
    ENERGY_LABELS,
    NL_LAT_MAX,
    NL_LAT_MIN,
    NL_LON_MAX,
    NL_LON_MIN,
    PROPERTY_TYPES,
    REGIONS,
)


@dataclass
class ValidationResult:
    is_valid: bool
    dq_flags: list[str] = field(default_factory=list)
    cleaned: dict[str, Any] = field(default_factory=dict)


def _flag(result: ValidationResult, flag: str) -> None:
    result.dq_flags.append(flag)


def validate_listing(raw: dict[str, Any], reference_date: date | None = None) -> ValidationResult:
    """Validate and clean a single listing record."""
    ref = reference_date or date.today()
    result = ValidationResult(is_valid=True)
    cleaned: dict[str, Any] = {}

    listing_id = raw.get("listing_id")
    if not listing_id:
        _flag(result, "missing_listing_id")
        result.is_valid = False
    cleaned["listing_id"] = str(listing_id) if listing_id else None

    for field_name in [
        "region",
        "postcode",
        "property_type",
        "energy_label",
    ]:
        val = raw.get(field_name)
        if val is None or (isinstance(val, str) and not val.strip()):
            _flag(result, f"missing_{field_name}")
        cleaned[field_name] = str(val).strip() if val is not None else None

    try:
        surface = float(raw["surface_area"])
        if surface <= 0:
            _flag(result, "invalid_surface_area")
            result.is_valid = False
        cleaned["surface_area"] = surface
    except (KeyError, TypeError, ValueError):
        _flag(result, "invalid_surface_area")
        result.is_valid = False
        cleaned["surface_area"] = None

    try:
        rooms = int(raw["number_of_rooms"])
        if rooms <= 0:
            _flag(result, "invalid_number_of_rooms")
            result.is_valid = False
        cleaned["number_of_rooms"] = rooms
    except (KeyError, TypeError, ValueError):
        _flag(result, "invalid_number_of_rooms")
        result.is_valid = False
        cleaned["number_of_rooms"] = None

    try:
        bedrooms = int(raw["number_of_bedrooms"])
        if bedrooms < 0:
            _flag(result, "invalid_number_of_bedrooms")
            result.is_valid = False
        cleaned["number_of_bedrooms"] = bedrooms
    except (KeyError, TypeError, ValueError):
        _flag(result, "invalid_number_of_bedrooms")
        result.is_valid = False
        cleaned["number_of_bedrooms"] = None

    try:
        build_year = int(raw["build_year"])
        if build_year > ref.year:
            _flag(result, "build_year_in_future")
            result.is_valid = False
        cleaned["build_year"] = build_year
    except (KeyError, TypeError, ValueError):
        _flag(result, "invalid_build_year")
        result.is_valid = False
        cleaned["build_year"] = None

    energy = cleaned.get("energy_label")
    if energy and energy not in ENERGY_LABELS:
        _flag(result, "unknown_energy_label")
        result.is_valid = False

    prop_type = cleaned.get("property_type")
    if prop_type and prop_type not in PROPERTY_TYPES:
        _flag(result, "unknown_property_type")

    region = cleaned.get("region")
    if region and region not in REGIONS:
        _flag(result, "unknown_region")

    try:
        lat = float(raw["latitude"])
        lon = float(raw["longitude"])
        if not (NL_LAT_MIN <= lat <= NL_LAT_MAX and NL_LON_MIN <= lon <= NL_LON_MAX):
            _flag(result, "coordinates_out_of_nl")
        cleaned["latitude"] = lat
        cleaned["longitude"] = lon
    except (KeyError, TypeError, ValueError):
        _flag(result, "invalid_coordinates")
        result.is_valid = False
        cleaned["latitude"] = None
        cleaned["longitude"] = None

    garden = raw.get("garden")
    cleaned["garden"] = bool(garden) if garden is not None else False

    for optional in ["asking_price", "sale_price", "sale_date", "listing_timestamp"]:
        cleaned[optional] = raw.get(optional)

    result.cleaned = cleaned
    return result


def validate_prediction_request(raw: dict[str, Any]) -> ValidationResult:
    """Validate an online prediction request (subset of listing fields)."""
    required = [
        "surface_area",
        "number_of_rooms",
        "number_of_bedrooms",
        "build_year",
        "energy_label",
        "property_type",
        "region",
        "latitude",
        "longitude",
    ]
    for key in required:
        if key not in raw or raw[key] is None:
            return ValidationResult(is_valid=False, dq_flags=[f"missing_{key}"])

    # Reuse listing validation with synthetic listing_id for field checks
    return validate_listing({**raw, "listing_id": raw.get("listing_id", "prediction-request")})
