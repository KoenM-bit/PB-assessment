# Databricks notebook source
# Train: log experiment + register UC version (no @challenger alias).

# COMMAND ----------

import os
from pathlib import Path

from house_price_ml.data.training_data import assemble_training_frame
from house_price_ml.models.train import train

catalog = dbutils.widgets.get("catalog") or "house_price_staging"
widget_commit = dbutils.widgets.get("git_commit")
if widget_commit and widget_commit not in ("unknown", "none", ""):
    os.environ["GIT_COMMIT"] = widget_commit

silver_df = spark.table(f"{catalog}.silver.listings_clean").toPandas()
gold_df = spark.table(f"{catalog}.gold.listing_features").toPandas()
training_df = assemble_training_frame(silver_df, gold_df)

out = Path("/tmp/model_output")
train(
    training_df,
    output_dir=out,
    catalog=catalog,
    register_model=True,
    git_commit=widget_commit if widget_commit not in ("unknown", "none", "") else None,
    data_source=f"{catalog}.gold.listing_features",
)

print("Training complete. See MLflow experiment /Shared/house_price_prediction")
print("Model registered without alias. To go live: promote-challenger, then deploy-serving-from-registry")
