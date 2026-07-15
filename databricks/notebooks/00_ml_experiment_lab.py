# Databricks notebook source
# MAGIC %md
# MAGIC # ML Experiment Lab
# MAGIC
# MAGIC Guided playbook for **explore → experiment → official CI** without registering Unity Catalog models.
# MAGIC
# MAGIC | Lane | UC register | MLflow experiment |
# MAGIC |------|-------------|-------------------|
# MAGIC | This notebook | No | `/Shared/house_price_prediction_lab` |
# MAGIC | `ml_experiment_pipeline` | No | `_lab` |
# MAGIC | `full_ml_pipeline` (push to staging) | Yes (if gates pass) | `/Shared/house_price_prediction` |

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Setup

# COMMAND ----------

import json
import os
from pathlib import Path

import mlflow
import pandas as pd
from mlflow import MlflowClient

from house_price_ml.config.training_config import load_training_config
from house_price_ml.evaluation.gate_config import load_quality_gates
from house_price_ml.evaluation.gates import evaluate_training_gates
from house_price_ml.evaluation.splits import holdout_test_split
from house_price_ml.jobs.databricks_train import apply_experiment_flags, parse_bool, training_config_from_job_params
from house_price_ml.jobs.experiment_lab import (
    LAB_MLFLOW_EXPERIMENT,
    baseline_holdout_metrics,
    configure_lab_mlflow,
    data_quality_summary,
    format_gate_report,
    load_training_frame_from_catalog,
    load_training_frame_sample,
    search_recent_runs,
    segment_balance_report,
)
from house_price_ml.models.train import train

# COMMAND ----------


def _widget(name: str, default: str = "") -> str:
    try:
        dbutils.widgets.text(name, default)
        return dbutils.widgets.get(name)
    except Exception:
        return default


dbutils.widgets.dropdown("catalog", "house_price_staging", ["house_price_staging", "house_price_prod"])
dbutils.widgets.dropdown("data_source", "sample", ["delta", "sample"])
dbutils.widgets.dropdown("sample_profile", "demo", ["demo", "engineering"])
dbutils.widgets.text("sample_rows", "")
dbutils.widgets.dropdown("enable_tuning", "false", ["true", "false"])
dbutils.widgets.dropdown("enable_ablation", "false", ["true", "false"])
dbutils.widgets.dropdown("enable_explainability", "false", ["true", "false"])
dbutils.widgets.text("mlflow_experiment", LAB_MLFLOW_EXPERIMENT)

catalog = _widget("catalog", "house_price_staging")
data_source = _widget("data_source", "sample")
sample_profile = _widget("sample_profile", "demo")
sample_rows_raw = _widget("sample_rows", "").strip()
sample_rows = int(sample_rows_raw) if sample_rows_raw else None
enable_tuning = parse_bool(_widget("enable_tuning", "false"))
enable_ablation = parse_bool(_widget("enable_ablation", "false"))
enable_explainability = parse_bool(_widget("enable_explainability", "false"))
mlflow_experiment = _widget("mlflow_experiment", LAB_MLFLOW_EXPERIMENT)

tracking_uri = configure_lab_mlflow(mlflow_experiment)
print("Lane contract: register_model=False — logs to MLflow lab experiment only (no UC version).")
print(f"MLflow experiment: {mlflow_experiment}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Load data

# COMMAND ----------

silver_df: pd.DataFrame | None = None
rejected_df: pd.DataFrame | None = None

if data_source == "delta":
    training_df = load_training_frame_from_catalog(spark, catalog)
    silver_df = spark.table(f"{catalog}.silver.listings_clean").toPandas()
    try:
        rejected_df = spark.table(f"{catalog}.silver.listings_rejected").toPandas()
    except Exception:
        rejected_df = None
    print(f"Loaded {len(training_df)} training rows from Delta ({catalog})")
else:
    training_df = load_training_frame_sample(profile=sample_profile, rows=sample_rows)
    print(f"Loaded {len(training_df)} in-memory sample rows (profile={sample_profile})")

display(training_df.head(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Data quality

# COMMAND ----------

if silver_df is not None:
    dq = data_quality_summary(silver_df, rejected_df)
else:
    dq = {
        "silver_rows": len(training_df),
        "rejected_rows": 0,
        "reject_rate": 0.0,
        "null_rates": data_quality_summary(training_df, None)["null_rates"],
    }

print(json.dumps(dq, indent=2))
if rejected_df is not None and len(rejected_df):
    display(rejected_df.head(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Segment explorer

# COMMAND ----------

gates_config = load_quality_gates()
segment_report = segment_balance_report(training_df)
display(segment_report)

min_n = gates_config.segments.min_segment_sample_size
thin = segment_report[segment_report["count"] < min_n]
if len(thin):
    print(f"Segments below min_segment_sample_size={min_n}:")
    display(thin)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Baseline check (holdout, no MLflow)

# COMMAND ----------

training_config = load_training_config()
train_df, test_df = holdout_test_split(
    training_df,
    test_quarters=training_config.splits.holdout_test_quarters,
)
baseline_metrics = baseline_holdout_metrics(train_df, test_df)
print(json.dumps(baseline_metrics, indent=2))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Quick train (log only)

# COMMAND ----------

quick_config = apply_experiment_flags(
    training_config,
    enable_tuning=False,
    enable_ablation=False,
    enable_explainability=False,
)

out = Path("/tmp/lab_model_output")
model_path = train(
    training_df,
    output_dir=out,
    training_config=quick_config,
    catalog=catalog,
    register_model=False,
    enforce_gates=False,
    data_source=f"lab:{data_source}",
    mlflow_experiment_name=mlflow_experiment,
)

recent = search_recent_runs(mlflow_experiment, n=1)
run_id = recent.iloc[0]["run_id"] if len(recent) else "unknown"
print(f"Run ID: {run_id}")
print(f"Artifacts: {model_path}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Gate drill-down

# COMMAND ----------

import tempfile

client = MlflowClient()
with tempfile.TemporaryDirectory() as tmp:
    gate_path = client.download_artifacts(run_id, "reports/gate_report.json", tmp)
    gate_payload = json.loads(Path(gate_path).read_text())

    from house_price_ml.evaluation.gates import GateResult
    from house_price_ml.evaluation.metrics import compute_metrics, evaluate_by_segment
    from house_price_ml.evaluation.segments import price_category

    gate_result = GateResult(
        passed=gate_payload.get("passed", False),
        failures=gate_payload.get("failures", []),
        details=gate_payload.get("details", {}),
    )
    display(format_gate_report(gate_result))

    what_if_gates = gates_config.model_copy(
        update={
            "segments": gates_config.segments.model_copy(
                update={"max_mae_degradation_vs_overall": 1.05}
            )
        }
    )
    try:
        holdout_path = client.download_artifacts(run_id, "reports/holdout_predictions.csv", tmp)
        holdout = pd.read_csv(holdout_path)
        overall_mae = compute_metrics(
            holdout["label_sale_price"].values.astype(float),
            holdout["predicted_price"].values.astype(float),
        )["mae"]
        seg_region = evaluate_by_segment(holdout, "label_sale_price", "predicted_price", "region")
        seg_prop = evaluate_by_segment(holdout, "label_sale_price", "predicted_price", "property_type")
        holdout["price_category"] = holdout["label_sale_price"].apply(price_category)
        seg_price = evaluate_by_segment(holdout, "label_sale_price", "predicted_price", "price_category")
        what_if_summary = {
            "test_metrics": {"mae": overall_mae, "pct_within_10pct": 0.65},
            "baseline_metrics": {"mae": overall_mae * 1.1},
            "beats_baseline": True,
            "walk_forward_model_mae_mean": overall_mae,
            "walk_forward_baseline_mae_mean": overall_mae * 1.1,
        }
        what_if_result = evaluate_training_gates(
            what_if_summary, seg_region, seg_prop, seg_price, gates=what_if_gates
        )
        print("What-if with max_mae_degradation_vs_overall=1.05 (not persisted):")
        display(format_gate_report(what_if_result))
    except Exception as exc:
        print(f"What-if skipped: {exc}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Deep experiment (tuning + ablation + SHAP)

# COMMAND ----------

deep_config = training_config_from_job_params(
    enable_tuning="true",
    enable_ablation="true",
    enable_explainability="true",
)

deep_out = Path("/tmp/lab_deep_output")
train(
    training_df,
    output_dir=deep_out,
    training_config=deep_config,
    catalog=catalog,
    register_model=False,
    enforce_gates=False,
    data_source=f"lab-deep:{data_source}",
    mlflow_experiment_name=mlflow_experiment,
)
print("Deep experiment complete (same flags as ml_experiment_pipeline, still no UC register).")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Compare recent runs

# COMMAND ----------

runs_df = search_recent_runs(mlflow_experiment, n=10)
display(runs_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. YAML export checklist
# MAGIC
# MAGIC Before pushing to `staging` (official `full_ml_pipeline`), commit:
# MAGIC
# MAGIC - `ml/config/training.yaml` — hyperparams; set `tuning.enabled: false` for prod CI
# MAGIC - `ml/config/quality_gates.yaml` — gate thresholds
# MAGIC - `ml/config/data.yaml` — if synthetic profile / data assumptions changed
# MAGIC - `ml/src/` — feature or training code changes
# MAGIC
# MAGIC After YAML is committed:
# MAGIC
# MAGIC ```bash
# MAGIC # Batch experiment (no UC register)
# MAGIC ./scripts/databricks-ci.sh run-experiment-pipeline staging
# MAGIC
# MAGIC # Official candidate (registers UC version when gates pass)
# MAGIC git push origin staging
# MAGIC ```

# COMMAND ----------

print("Lab playbook complete. Promote only from official experiment runs with gates_passed=1.")
