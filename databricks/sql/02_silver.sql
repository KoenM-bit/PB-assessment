-- Silver layer: validated listings
CREATE SCHEMA IF NOT EXISTS ${catalog}.silver;

CREATE TABLE IF NOT EXISTS ${catalog}.silver.listings_clean (
    listing_id STRING NOT NULL,
    listing_timestamp TIMESTAMP,
    region STRING,
    postcode STRING,
    latitude DOUBLE,
    longitude DOUBLE,
    surface_area DOUBLE,
    number_of_rooms INT,
    number_of_bedrooms INT,
    build_year INT,
    energy_label STRING,
    property_type STRING,
    garden BOOLEAN,
    asking_price DOUBLE,
    sale_price DOUBLE,
    sale_date DATE,
    is_duplicate BOOLEAN,
    dq_flags ARRAY<STRING>,
    is_valid BOOLEAN,
    cleaned_at TIMESTAMP
)
USING DELTA;

CREATE TABLE IF NOT EXISTS ${catalog}.silver.listings_rejected (
    listing_id STRING,
    dq_flags ARRAY<STRING>,
    rejected_payload STRING,
    rejected_at TIMESTAMP
)
USING DELTA;
