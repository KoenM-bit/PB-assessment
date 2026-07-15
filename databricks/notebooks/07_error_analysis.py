# Databricks notebook source
# Error analysis on the latest training run that passed quality gates.

# COMMAND ----------

import json
import tempfile
from pathlib import Path

import mlflow
import pandas as pd
from mlflow import MlflowClient

from house_price_ml.evaluation.metrics import compute_metrics, evaluate_by_segment
from house_price_ml.evaluation.segments import price_category

# COMMAND ----------


def _widget(name: str, default: str = "") -> str:
    try:
        return dbutils.widgets.get(name)
    except Exception:
        return default


catalog = _widget("catalog", "house_price_staging")
experiment_name = _widget("mlflow_experiment", "/Shared/house_price_prediction")

mlflow.set_tracking_uri("databricks")
client = MlflowClient()

experiment = client.get_experiment_by_name(experiment_name)
if experiment is None:
    dbutils.notebook.exit(f"Experiment not found: {experiment_name}")

runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    filter_string="metrics.gates_passed = 1",
    order_by=["start_time DESC"],
    max_results=1,
)
if not runs:
    dbutils.notebook.exit("No training runs with gates_passed=1 found")

run = runs[0]
run_id = run.info.run_id
print(f"Analyzing MLflow run {run_id} ({run.info.run_name})")

artifact_path = client.download_artifacts(run_id, "reports/holdout_predictions.csv")
holdout = pd.read_csv(artifact_path)
if "label_sale_price" not in holdout.columns:
    dbutils.notebook.exit("holdout_predictions.csv missing label_sale_price column")

holdout["price_category"] = holdout["label_sale_price"].apply(price_category)
overall = compute_metrics(
    holdout["label_sale_price"].values.astype(float),
    holdout["predicted_price"].values.astype(float),
)

segment_tables = {
    "region": evaluate_by_segment(holdout, "label_sale_price", "predicted_price", "region"),
    "property_type": evaluate_by_segment(
        holdout, "label_sale_price", "predicted_price", "property_type"
    ),
    "price_category": evaluate_by_segment(
        holdout, "label_sale_price", "predicted_price", "price_category"
    ),
}

residual_summary = {
    "mlflow_run_id": run_id,
    "run_name": run.info.run_name,
    "overall_metrics": overall,
    "residual_mean": float(holdout["residual"].mean()) if "residual" in holdout else None,
    "residual_std": float(holdout["residual"].std()) if "residual" in holdout else None,
    "segments": {name: df.to_dict(orient="records") for name, df in segment_tables.items()},
}

out_table = f"{catalog}.gold.error_analysis_summary"
summary_df = spark.createDataFrame(
    [
        {
            "mlflow_run_id": run_id,
            "run_name": run.info.run_name,
            "overall_mae": overall["mae"],
            "overall_pct_within_10pct": overall["pct_within_10pct"],
            "summary_json": json.dumps(residual_summary),
        }
    ]
)
summary_df.write.format("delta").mode("append").option("mergeSchema", "true").saveAsTable(out_table)
print(f"Wrote error analysis summary to {out_table}")

with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
    json.dump(residual_summary, handle, indent=2)
    temp_path = handle.name

client.log_artifact(run_id, temp_path, artifact_path="error_analysis")
Path(temp_path).unlink(missing_ok=True)
print(f"Logged error_analysis artifact to MLflow run {run_id}")
