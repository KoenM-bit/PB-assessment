-- Bronze layer: raw listings
CREATE SCHEMA IF NOT EXISTS ${catalog}.bronze;

CREATE TABLE IF NOT EXISTS ${catalog}.bronze.listings_raw (
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
    ingestion_timestamp TIMESTAMP,
    ingestion_date DATE,
    source_file STRING,
    raw_payload STRING
)
USING DELTA
PARTITIONED BY (ingestion_date);
