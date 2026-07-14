# Databricks notebook source
# Bronze ingest
# Thin wrapper — logic lives in house_price_ml package
# Wheel is installed via serverless environment dependencies (databricks.yml).

# COMMAND ----------

from house_price_ml.data.synthetic import generate_listings

catalog = dbutils.widgets.get("catalog") or "house_price_staging"
try:
    n_rows = int(dbutils.widgets.get("rows"))
except Exception:
    n_rows = 500

df = generate_listings(n_rows)
df["source_file"] = "synthetic"
df["raw_payload"] = None
spark.createDataFrame(df).write.format("delta").mode("append").option(
    "mergeSchema", "true"
).saveAsTable(f"{catalog}.bronze.listings_raw")
