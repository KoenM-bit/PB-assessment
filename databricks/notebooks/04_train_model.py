# Databricks notebook source
# Train, log to MLflow Experiments, and register in Unity Catalog.

# COMMAND ----------

from pathlib import Path

from house_price_ml.models.train import train

catalog = dbutils.widgets.get("catalog") or "house_price_staging"
model_alias = dbutils.widgets.get("model_alias") or "challenger"

gold_df = spark.table(f"{catalog}.gold.listing_features").toPandas()
silver_df = spark.table(f"{catalog}.silver.listings_clean").toPandas()
merged = silver_df.merge(gold_df, on="listing_id", suffixes=("", "_gold"))
tmp_path = "/tmp/train_data.csv"
merged.to_csv(tmp_path, index=False)

out = Path("/tmp/model_output")
train(
    Path(tmp_path),
    "random_forest",
    out,
    catalog=catalog,
    model_alias=model_alias,
)

print(f"Training complete. See MLflow experiment /Shared/house_price_prediction")
print(f"Model alias {model_alias} updated in {catalog}.gold.house_price_model")
