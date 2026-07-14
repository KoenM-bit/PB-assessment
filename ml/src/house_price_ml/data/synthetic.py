"""Synthetic Dutch housing dataset generator."""

from __future__ import annotations

import argparse
import uuid
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from house_price_ml.config.constants import CITY_CENTRES, ENERGY_LABELS, PROPERTY_TYPES, REGIONS


def _base_price(region: str, prop_type: str, surface: float) -> float:
    region_factor = {
        "Amsterdam": 1.35,
        "Rotterdam": 1.05,
        "Utrecht": 1.25,
        "The Hague": 1.15,
        "Eindhoven": 0.95,
        "Groningen": 0.85,
        "Maastricht": 0.90,
        "Nijmegen": 0.88,
    }.get(region, 1.0)
    type_factor = {
        "apartment": 1.1,
        "terraced_house": 1.0,
        "semi_detached": 1.15,
        "detached": 1.3,
        "bungalow": 1.05,
    }.get(prop_type, 1.0)
    return region_factor * type_factor * surface * 3200


def generate_listings(n: int = 500, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    start = date(2023, 1, 1)

    for i in range(n):
        region = rng.choice(REGIONS)
        prop_type = rng.choice(PROPERTY_TYPES)
        surface = float(rng.integers(45, 220))
        rooms = max(2, int(surface // 25) + rng.integers(-1, 2))
        bedrooms = max(1, rooms - rng.integers(1, 3))
        build_year = int(rng.integers(1950, 2024))
        energy = rng.choice(ENERGY_LABELS, p=[0.02, 0.05, 0.1, 0.2, 0.2, 0.15, 0.12, 0.1, 0.06])
        garden = bool(rng.random() > 0.3)
        listing_date = start + timedelta(days=int(rng.integers(0, 900)))
        noise = rng.normal(0, 35000)
        sale_price = max(150000, _base_price(region, prop_type, surface) + noise)
        # Non-linear effects the business baseline does not model
        age = listing_date.year - build_year
        sale_price += max(0, (30 - age)) * 1500  # newer homes premium
        sale_price += (rooms - 4) * 8000  # room count effect
        if energy in ("A++", "A+", "A"):
            sale_price *= 1.08
        elif energy in ("F", "G"):
            sale_price *= 0.92
        if garden:
            sale_price *= 1.05
        sale_price += rng.normal(0, 15000)  # extra noise

        centre_lat, centre_lon = CITY_CENTRES[region]
        # Jitter around the regional city centre (~0–15 km) so geo features match the region label.
        latitude = centre_lat + rng.normal(0, 0.04)
        longitude = centre_lon + rng.normal(0, 0.06)

        rows.append(
            {
                "listing_id": str(uuid.uuid4()),
                "listing_timestamp": datetime.combine(listing_date, datetime.min.time()),
                "region": region,
                "postcode": f"{rng.integers(1000, 9999)} {rng.choice(['AA','AB','CD','EF'])}",
                "latitude": latitude,
                "longitude": longitude,
                "surface_area": surface,
                "number_of_rooms": rooms,
                "number_of_bedrooms": bedrooms,
                "build_year": build_year,
                "energy_label": energy,
                "property_type": prop_type,
                "garden": garden,
                "asking_price": sale_price * rng.uniform(0.95, 1.08),
                "sale_price": sale_price,
                "sale_date": listing_date + timedelta(days=int(rng.integers(14, 120))),
                "ingestion_timestamp": datetime.utcnow(),
                "ingestion_date": datetime.utcnow().date(),
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    df = generate_listings(args.rows, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"Wrote {len(df)} listings to {args.output}")


if __name__ == "__main__":
    main()
