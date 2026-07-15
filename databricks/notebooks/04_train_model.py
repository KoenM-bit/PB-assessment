# Databricks notebook source
# Train: log experiment + register UC version (no @challenger alias).

# COMMAND ----------

import os
from pathlib import Path

from house_price_ml.data.training_data import assemble_training_frame
from house_price_ml.models.train import train

catalog = dbutils.widgets.get("catalog") or "house_price_staging"
try:
    widget_commit = dbutils.widgets.get("git_commit")
except Exception:
    widget_commit = ""
if widget_commit and widget_commit not in ("unknown", "none", ""):
    os.environ["GIT_COMMIT"] = widget_commit

silver_table = f"{catalog}.silver.listings_clean"
gold_table = f"{catalog}.gold.listing_features"


def latest_delta_version(table: str) -> int:
    row = spark.sql(f"DESCRIBE HISTORY {table} LIMIT 1").first()
    return int(row.version)


table_versions = {
    silver_table: latest_delta_version(silver_table),
    gold_table: latest_delta_version(gold_table),
}

silver_df = spark.table(silver_table).toPandas()
gold_df = spark.table(gold_table).toPandas()
training_df = assemble_training_frame(silver_df, gold_df)

out = Path("/tmp/model_output")
train(
    training_df,
    output_dir=out,
    catalog=catalog,
    register_model=True,
    git_commit=widget_commit if widget_commit not in ("unknown", "none", "") else None,
    data_source=gold_table,
    table_versions=table_versions,
)

print("Training complete. See MLflow experiment /Shared/house_price_prediction")
print("Model registered without alias. To go live: promote-challenger, then deploy-serving-from-registry")
