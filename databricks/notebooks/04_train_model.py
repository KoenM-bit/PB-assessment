# Databricks notebook source
# Train and log to MLflow Experiments (no alias — promote separately).

# COMMAND ----------

from pathlib import Path

from house_price_ml.models.train import train

catalog = dbutils.widgets.get("catalog") or "house_price_staging"
git_commit = dbutils.widgets.get("git_commit") or None

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
    git_commit=git_commit,
    data_source=f"{catalog}.gold.listing_features",
)

print("Training complete. See MLflow experiment /Shared/house_price_prediction")
print("To make a run live in staging: promote-challenger with the run ID, then deploy-serving-from-registry")
