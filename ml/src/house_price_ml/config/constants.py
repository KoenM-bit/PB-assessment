"""Domain constants and allowed values."""

from __future__ import annotations

ENERGY_LABELS = ["A++", "A+", "A", "B", "C", "D", "E", "F", "G"]

ENERGY_LABEL_SCORES: dict[str, int] = {
    "A++": 10,
    "A+": 9,
    "A": 8,
    "B": 7,
    "C": 6,
    "D": 5,
    "E": 4,
    "F": 3,
    "G": 2,
}

PROPERTY_TYPES = [
    "apartment",
    "terraced_house",
    "semi_detached",
    "detached",
    "bungalow",
]

REGIONS = [
    "Amsterdam",
    "Rotterdam",
    "Utrecht",
    "The Hague",
    "Eindhoven",
    "Groningen",
    "Maastricht",
    "Nijmegen",
]

# Approximate city centre coordinates for distance features
CITY_CENTRES: dict[str, tuple[float, float]] = {
    "Amsterdam": (52.3676, 4.9041),
    "Rotterdam": (51.9244, 4.4777),
    "Utrecht": (52.0907, 5.1214),
    "The Hague": (52.0705, 4.3007),
    "Eindhoven": (51.4416, 5.4697),
    "Groningen": (53.2194, 6.5665),
    "Maastricht": (50.8514, 5.6910),
    "Nijmegen": (51.8426, 5.8528),
}

# Netherlands bounding box (approximate)
NL_LAT_MIN, NL_LAT_MAX = 50.75, 53.55
NL_LON_MIN, NL_LON_MAX = 3.35, 7.22

PRICE_CATEGORY_BOUNDS = [
    (0, 250_000, "low"),
    (250_000, 400_000, "medium"),
    (400_000, 600_000, "high"),
    (600_000, float("inf"), "premium"),
]
