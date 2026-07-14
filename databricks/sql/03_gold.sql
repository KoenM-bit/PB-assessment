-- Gold layer: features, predictions, monitoring
CREATE SCHEMA IF NOT EXISTS ${catalog}.gold;

CREATE TABLE IF NOT EXISTS ${catalog}.gold.listing_features (
    listing_id STRING NOT NULL,
    feature_snapshot_date DATE,
    house_age DOUBLE,
    surface_per_room DOUBLE,
    energy_label_score INT,
    surface_x_energy DOUBLE,
    dist_to_city_centre_km DOUBLE,
    region_median_price_per_sqm DOUBLE,
    month INT,
    quarter INT,
    label_sale_price DOUBLE
)
USING DELTA;

CREATE TABLE IF NOT EXISTS ${catalog}.gold.predictions (
    prediction_id STRING NOT NULL,
    listing_id STRING,
    address STRING,
    postcode STRING,
    property_key STRING,
    request_payload STRING,
    predicted_price DOUBLE,
    model_name STRING,
    model_version STRING,
    model_alias STRING,
    prediction_timestamp TIMESTAMP,
    app_env STRING,
    serving_latency_ms INT,
    warnings ARRAY<STRING>,
    is_fallback BOOLEAN
)
USING DELTA;

CREATE TABLE IF NOT EXISTS ${catalog}.gold.actual_sales (
    actual_sale_id STRING NOT NULL,
    prediction_id STRING,
    listing_id STRING,
    actual_sale_price DOUBLE,
    sale_date DATE,
    recorded_at TIMESTAMP,
    recorded_by STRING
)
USING DELTA;

CREATE TABLE IF NOT EXISTS ${catalog}.gold.model_evaluations (
    evaluation_id STRING NOT NULL,
    evaluation_date DATE,
    window_type STRING,
    segment_type STRING,
    segment_value STRING,
    sample_size INT,
    mae DOUBLE,
    rmse DOUBLE,
    bias DOUBLE,
    mape DOUBLE,
    model_version STRING
)
USING DELTA;

CREATE TABLE IF NOT EXISTS ${catalog}.gold.feature_monitoring (
    monitoring_date DATE,
    feature_name STRING,
    reference_mean DOUBLE,
    reference_std DOUBLE,
    recent_mean DOUBLE,
    recent_std DOUBLE,
    pct_out_of_range DOUBLE,
    drift_score DOUBLE,
    sample_size INT
)
USING DELTA;

CREATE TABLE IF NOT EXISTS ${catalog}.gold.serving_metrics (
    date DATE,
    request_count INT,
    error_count INT,
    timeout_count INT,
    p50_latency_ms DOUBLE,
    p95_latency_ms DOUBLE
)
USING DELTA;
