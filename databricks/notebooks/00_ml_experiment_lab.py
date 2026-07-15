# Databricks notebook source
# MAGIC %md
# MAGIC # ML Experiment Lab
# MAGIC
# MAGIC **Playbook:** Data quality → EDA → feature matrix → model selection → train → gates → residuals → SHAP
# MAGIC
# MAGIC Edit **`ml/config/eda_lab.yaml`** to toggle sections, hypotheses, and feature-matrix experiments.
# MAGIC
# MAGIC | Lane | UC register | MLflow experiment |
# MAGIC |------|-------------|-------------------|
# MAGIC | This notebook | No | `/Shared/house_price_prediction_lab` |
# MAGIC | `ml_experiment_pipeline` | No | `_lab` |
# MAGIC | `full_ml_pipeline` | Yes (if gates pass) | `/Shared/house_price_prediction` |

# COMMAND ----------

# MAGIC %md
# MAGIC ## 0. Install package (interactive notebooks)
# MAGIC
# MAGIC Jobs attach the wheel automatically; **interactive** runs need a fresh wheel after deploy.
# MAGIC
# MAGIC 1. Run `make databricks-bundle-deploy` (or wait for CI `bundle-deploy`).
# MAGIC 2. Set **`wheel_path`** to your bundle artifact, e.g.  
# MAGIC    `/Workspace/Users/<you>/.bundle/house-price-ml/staging/artifacts/.internal/house_price_ml-0.1.0-py3-none-any.whl`
# MAGIC 3. Run this cell twice if prompted (install → Python restart → verify).

# COMMAND ----------


def _widget(name: str, default: str = "") -> str:
    try:
        dbutils.widgets.text(name, default)
        return dbutils.widgets.get(name)
    except Exception:
        return default


dbutils.widgets.text("wheel_path", "")

import importlib.util
import subprocess
import sys

_wheel_default_help = (
    "Set wheel_path widget to the bundle wheel (.whl), then re-run this cell. "
    "Example: /Workspace/Users/<you>/.bundle/house-price-ml/staging/artifacts/.internal/"
    "house_price_ml-0.1.0-py3-none-any.whl"
)

if importlib.util.find_spec("house_price_ml.config.eda_lab_config") is None:
    wheel_path = _widget("wheel_path", "").strip()
    if not wheel_path:
        raise ModuleNotFoundError(
            "house_price_ml.config.eda_lab_config not found (stale wheel). " + _wheel_default_help
        )
    print(f"Installing wheel from {wheel_path}")
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", wheel_path, "--force-reinstall", "-q"]
    )
    dbutils.library.restartPython()
else:
    import house_price_ml

    print(f"house_price_ml OK — version {getattr(house_price_ml, '__version__', '0.1.0')}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Setup

# COMMAND ----------

import json
import os
import tempfile
from pathlib import Path

import mlflow
import pandas as pd
from mlflow import MlflowClient

from house_price_ml.config.eda_lab_config import load_eda_lab_config
from house_price_ml.config.training_config import load_training_config
from house_price_ml.evaluation.gates import GateResult, evaluate_training_gates
from house_price_ml.evaluation.metrics import compute_metrics, evaluate_by_segment
from house_price_ml.evaluation.segments import price_category
from house_price_ml.evaluation.splits import holdout_test_split
from house_price_ml.jobs.databricks_train import apply_experiment_flags, parse_bool, training_config_from_job_params
from house_price_ml.jobs.eda_lab import (
    bivariate_vs_target,
    correlation_report,
    evaluate_business_hypotheses,
    evaluate_data_quality_gates,
    residual_analysis_report,
    run_feature_matrix,
    run_model_selection,
    run_shap_report,
    univariate_profile,
)
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

dbutils.widgets.dropdown("catalog", "house_price_staging", ["house_price_staging", "house_price_prod"])
dbutils.widgets.dropdown("data_source", "sample", ["delta", "sample"])
dbutils.widgets.dropdown("sample_profile", "demo", ["demo", "engineering"])
dbutils.widgets.text("sample_rows", "")
dbutils.widgets.dropdown("enable_tuning", "false", ["true", "false"])
dbutils.widgets.dropdown("enable_ablation", "false", ["true", "false"])
dbutils.widgets.dropdown("enable_explainability", "false", ["true", "false"])
dbutils.widgets.text("mlflow_experiment", LAB_MLFLOW_EXPERIMENT)
dbutils.widgets.text("eda_config_path", "")

catalog = _widget("catalog", "house_price_staging")
data_source = _widget("data_source", "sample")
sample_profile = _widget("sample_profile", "demo")
sample_rows_raw = _widget("sample_rows", "").strip()
sample_rows = int(sample_rows_raw) if sample_rows_raw else None
enable_tuning = parse_bool(_widget("enable_tuning", "false"))
enable_ablation = parse_bool(_widget("enable_ablation", "false"))
enable_explainability = parse_bool(_widget("enable_explainability", "false"))
mlflow_experiment = _widget("mlflow_experiment", LAB_MLFLOW_EXPERIMENT)
eda_config_path = _widget("eda_config_path", "").strip() or None

eda_config = load_eda_lab_config(eda_config_path)
training_config = load_training_config()

tracking_uri = configure_lab_mlflow(mlflow_experiment)
print("Lane contract: register_model=False — logs to MLflow lab experiment only (no UC version).")
print(f"MLflow experiment: {mlflow_experiment}")
print(f"EDA config: {eda_config_path or 'ml/config/eda_lab.yaml (default)'}")

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
# MAGIC
# MAGIC Thresholds from `eda_lab.yaml` → `data_quality`.

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
if eda_config.sections.data_quality:
    display(evaluate_data_quality_gates(dq, eda_config))
if rejected_df is not None and len(rejected_df):
    display(rejected_df.head(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Univariate EDA
# MAGIC
# MAGIC Distributions, skew warnings, IQR outlier rates (`eda_lab.yaml` → `univariate`).

# COMMAND ----------

if eda_config.sections.univariate:
    uni = univariate_profile(training_df, eda_config)
    display(uni)
    skew_warn = uni[uni["skew_warning"] == True]  # noqa: E712
    if len(skew_warn):
        print("Columns with high skew:")
        display(skew_warn[["column", "skew", "outlier_rate"]])

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Bivariate EDA
# MAGIC
# MAGIC Numeric correlations + categorical mean deltas vs `label_sale_price`.

# COMMAND ----------

if eda_config.sections.bivariate:
    display(bivariate_vs_target(training_df, eda_config))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Correlation matrix
# MAGIC
# MAGIC Multicollinearity flags + weak target signals (`eda_lab.yaml` → `correlation`).

# COMMAND ----------

if eda_config.sections.correlation:
    corr_matrix, corr_flags = correlation_report(training_df, eda_config)
    if not corr_matrix.empty:
        display(corr_matrix)
    if not corr_flags.empty:
        print("Flagged pairs:")
        display(corr_flags)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Business hypotheses
# MAGIC
# MAGIC Edit `business_hypotheses` in `eda_lab.yaml` to add domain checks.

# COMMAND ----------

if eda_config.sections.business_hypotheses:
    display(evaluate_business_hypotheses(training_df, eda_config))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Segment balance
# MAGIC
# MAGIC Compare segment counts vs `quality_gates.yaml` `min_segment_sample_size`.

# COMMAND ----------

from house_price_ml.evaluation.gate_config import load_quality_gates

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
# MAGIC ## 9. Feature matrix (config-driven)
# MAGIC
# MAGIC Play with `feature_matrix.experiments` in `eda_lab.yaml` — each row trains a variant on holdout.

# COMMAND ----------

train_df, test_df = holdout_test_split(
    training_df,
    test_quarters=training_config.splits.holdout_test_quarters,
)

if eda_config.sections.feature_matrix and eda_config.feature_matrix.enabled:
    feature_matrix = run_feature_matrix(train_df, test_df, eda_config, training_config)
    display(feature_matrix.sort_values("mae"))
else:
    print("Feature matrix disabled in eda_lab.yaml")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Model selection (quick compare)
# MAGIC
# MAGIC Candidates from `eda_lab.yaml` → `model_selection.candidates`.

# COMMAND ----------

if eda_config.sections.model_selection and eda_config.model_selection.enabled:
    display(run_model_selection(train_df, test_df, eda_config, training_config))
else:
    print("Model selection disabled in eda_lab.yaml")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 11. Baseline holdout (no MLflow)

# COMMAND ----------

print(json.dumps(baseline_holdout_metrics(train_df, test_df), indent=2))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 12. Quick train (log only, lab MLflow)

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
# MAGIC ## 13. Gate drill-down

# COMMAND ----------

client = MlflowClient()
with tempfile.TemporaryDirectory() as tmp:
    gate_path = client.download_artifacts(run_id, "reports/gate_report.json", tmp)
    gate_payload = json.loads(Path(gate_path).read_text())

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
# MAGIC ## 14. Residual analysis

# COMMAND ----------

if eda_config.sections.residual_analysis:
    try:
        with tempfile.TemporaryDirectory() as tmp:
            holdout_path = client.download_artifacts(run_id, "reports/holdout_predictions.csv", tmp)
            holdout_df = pd.read_csv(holdout_path)
        seg_residuals, worst = residual_analysis_report(holdout_df, eda_config)
        print("Residuals by segment:")
        display(seg_residuals)
        print("Worst predictions:")
        display(worst)
    except Exception as exc:
        print(f"Residual analysis skipped: {exc}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 15. SHAP / feature importance
# MAGIC
# MAGIC Lightweight SHAP on holdout sample (`eda_lab.yaml` → `shap`). Requires `shap` on cluster.

# COMMAND ----------

if eda_config.sections.shap and eda_config.shap.enabled:
    shap_table = run_shap_report(train_df, test_df, eda_config, training_config)
    display(shap_table)
else:
    print("SHAP disabled in eda_lab.yaml")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 16. Deep experiment (tuning + ablation + SHAP via train job)

# COMMAND ----------

deep_config = training_config_from_job_params(
    enable_tuning="true" if enable_tuning else "false",
    enable_ablation="true" if enable_ablation else "false",
    enable_explainability="true" if enable_explainability else "false",
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
print("Deep experiment complete (still no UC register).")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 17. Compare recent runs

# COMMAND ----------

display(search_recent_runs(mlflow_experiment, n=10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 18. YAML export checklist
# MAGIC
# MAGIC Before official CI (`git push origin staging`), commit:
# MAGIC
# MAGIC - `ml/config/eda_lab.yaml` — EDA playbook + feature matrix experiments
# MAGIC - `ml/config/training.yaml` — hyperparams; `tuning.enabled: false` for prod CI
# MAGIC - `ml/config/quality_gates.yaml` — gate thresholds
# MAGIC - `ml/config/data.yaml` — if data profile changed
# MAGIC - `ml/src/` — feature engineering code changes
# MAGIC
# MAGIC ```bash
# MAGIC ./scripts/databricks-ci.sh run-experiment-pipeline staging   # batch lab lane
# MAGIC git push origin staging                                        # official CI
# MAGIC ```

# COMMAND ----------

print("Lab playbook complete. Promote only from official experiment runs with gates_passed=1.")
