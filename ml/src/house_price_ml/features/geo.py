"""Geographic feature calculations."""

from __future__ import annotations

import math

from house_price_ml.config.constants import CITY_CENTRES


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate great-circle distance in kilometres."""
    r = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def distance_to_city_centre(region: str, latitude: float, longitude: float) -> float:
    """Distance from listing to regional city centre."""
    centre = CITY_CENTRES.get(region)
    if centre is None:
        return 0.0
    return haversine_km(latitude, longitude, centre[0], centre[1])
