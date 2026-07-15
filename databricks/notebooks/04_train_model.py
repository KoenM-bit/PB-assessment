# Databricks notebook source
# Train: log experiment + register UC version (no @challenger alias).
# Job widgets: enable_tuning, enable_ablation, enable_explainability (true/false).

# COMMAND ----------

import os
from pathlib import Path

from house_price_ml.data.training_data import assemble_training_frame
from house_price_ml.jobs.databricks_train import parse_bool, training_config_from_job_params
from house_price_ml.models.train import train

# COMMAND ----------


def _widget(name: str, default: str = "") -> str:
    try:
        return dbutils.widgets.get(name)
    except Exception:
        return default


catalog = _widget("catalog", "house_price_staging")
widget_commit = _widget("git_commit")
enable_tuning = _widget("enable_tuning", "false")
enable_ablation = _widget("enable_ablation", "false")
enable_explainability = _widget("enable_explainability", "false")
model_type = _widget("model_type", "")
register_model = parse_bool(_widget("register_model", "true"), default=True)
mlflow_experiment = _widget("mlflow_experiment", "").strip() or None

if widget_commit and widget_commit not in ("unknown", "none", ""):
    os.environ["GIT_COMMIT"] = widget_commit

training_config = training_config_from_job_params(
    model_type=model_type or None,
    enable_tuning=enable_tuning,
    enable_ablation=enable_ablation,
    enable_explainability=enable_explainability,
)

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

print(
    f"Training rows={len(training_df)} | tuning={training_config.tuning.enabled} "
    f"| ablation={training_config.ablation.enabled} "
    f"| explainability={training_config.explainability.enabled} "
    f"| register={register_model} "
    f"| mlflow_experiment={mlflow_experiment or '(default)'}"
)

out = Path("/tmp/model_output")
train(
    training_df,
    output_dir=out,
    training_config=training_config,
    catalog=catalog,
    register_model=register_model,
    git_commit=widget_commit if widget_commit not in ("unknown", "none", "") else None,
    data_source=gold_table,
    table_versions=table_versions,
    mlflow_experiment_name=mlflow_experiment,
)

lane = "official UC register" if register_model else "experiment (log only, no UC)"
print(f"Training complete — {lane}. See MLflow experiment for run details.")
if register_model:
    print("Model registered without alias. To go live: promote-challenger, then deploy-serving-from-registry")
