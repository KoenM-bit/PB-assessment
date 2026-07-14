# Databricks notebook: Bronze ingest
# Thin wrapper — logic lives in house_price_ml package

# MAGIC %pip install /Workspace/Shared/house_price_ml-0.1.0-py3-none-any.whl

# COMMAND ----------

from house_price_ml.data.synthetic import generate_listings

catalog = dbutils.widgets.get("catalog") if dbutils.widgets.get("catalog") else "house_price_staging"
n_rows = int(dbutils.widgets.get("rows") or "500")

df = generate_listings(n_rows)
df["source_file"] = "synthetic"
df["raw_payload"] = None
df.write.format("delta").mode("append").saveAsTable(f"{catalog}.bronze.listings_raw")
