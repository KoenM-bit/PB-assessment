# Databricks notebook source
# Bronze ingest
# Thin wrapper — logic lives in house_price_ml package
# Wheel is installed via serverless environment dependencies (databricks.yml).

# COMMAND ----------

from house_price_ml.data.data_config import load_data_profile
from house_price_ml.data.synthetic import generate_listings

catalog = dbutils.widgets.get("catalog") or "house_price_staging"
profile_name = dbutils.widgets.get("profile") or None
try:
    n_rows = int(dbutils.widgets.get("rows"))
except Exception:
    n_rows = None

if profile_name or n_rows is None:
    profile = load_data_profile(profile_name or "engineering")
    df = generate_listings(
        n_rows or profile.rows,
        profile.seed,
        missing_rate=profile.missing_rate,
        outlier_rate=profile.outlier_rate,
        invalid_rate=profile.invalid_rate,
        start_year=profile.start_year,
        span_days=profile.span_days,
    )
else:
    df = generate_listings(n_rows)

df["source_file"] = "synthetic"
df["raw_payload"] = None
spark.createDataFrame(df).write.format("delta").mode("append").option(
    "mergeSchema", "true"
).saveAsTable(f"{catalog}.bronze.listings_raw")
