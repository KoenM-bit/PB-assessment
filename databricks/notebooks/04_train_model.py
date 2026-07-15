# Databricks notebook source
# Train: log experiment + register UC version (no @challenger alias).

# COMMAND ----------

import os
from pathlib import Path

from house_price_ml.models.train import train

catalog = dbutils.widgets.get("catalog") or "house_price_staging"
widget_commit = dbutils.widgets.get("git_commit")
if widget_commit and widget_commit not in ("unknown", "none", ""):
    os.environ["GIT_COMMIT"] = widget_commit

gold_df = spark.table(f"{catalog}.gold.listing_features").toPandas()
silver_df = spark.table(f"{catalog}.silver.listings_clean").toPandas()
merged = silver_df.merge(gold_df, on="listing_id", suffixes=("", "_gold"))
tmp_path = "/tmp/train_data.csv"
merged.to_csv(tmp_path, index=False) ## todo: maak de train functie zo dat het de data direct als dataframe meegeeft

out = Path("/tmp/model_output")
train(
    Path(tmp_path),
    "random_forest",
    out,
    catalog=catalog,
    git_commit=widget_commit if widget_commit not in ("unknown", "none", "") else None,
    data_source=f"{catalog}.gold.listing_features",
)

print("Training complete. See MLflow experiment /Shared/house_price_prediction")
print("Model registered without alias. To go live: promote-challenger, then deploy-serving-from-registry")
