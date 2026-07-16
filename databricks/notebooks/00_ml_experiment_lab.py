# Databricks notebook source
# MAGIC %md
# MAGIC # ML Experiment Lab — Enterprise Bootstrap
# MAGIC
# MAGIC **Lifecycle:** environment → config → raw/model data → data quality → dynamic EDA →
# MAGIC chronological train/validation/test → business baseline → single-feature screen →
# MAGIC generated feature matrix → stability analysis → model-family selection →
# MAGIC final holdout evaluation → residuals → SHAP → gates → MLflow logging.
# MAGIC
# MAGIC **Lane contract**
# MAGIC
# MAGIC - This notebook never registers a model in Unity Catalog.
# MAGIC - Feature and model selection happen on the validation period.
# MAGIC - The final test period is evaluated once for the selected candidate.
# MAGIC - Promotion remains the responsibility of the official production pipeline.
# MAGIC
# MAGIC Configuration is loaded from `ml/config/eda_lab_enterprise.yaml`.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 0. Interactive wheel installation

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

PACKAGE_CONFIG_MODULE = "house_price_ml.config.training_config"

if importlib.util.find_spec(PACKAGE_CONFIG_MODULE) is None:
    wheel_path = _widget("wheel_path", "").strip()
    if not wheel_path:
        raise ModuleNotFoundError(
            f"{PACKAGE_CONFIG_MODULE} is unavailable. "
            "Set the wheel_path widget to a deployed .whl and rerun this cell."
        )

    print(f"Installing wheel from: {wheel_path}")
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            wheel_path,
            "--force-reinstall",
            "-q",
        ]
    )
    dbutils.library.restartPython()
else:
    import house_price_ml

    print(
        "house_price_ml available — "
        f"version={getattr(house_price_ml, '__version__', 'unknown')}"
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Imports, widgets, and lane contract

# COMMAND ----------

import itertools
import json
import math
import os
import tempfile
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import mlflow
import numpy as np
import pandas as pd
from mlflow import MlflowClient

import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

try:
    from catboost import CatBoostRegressor
except Exception:
    CatBoostRegressor = None

from house_price_ml.config.training_config import load_training_config
from house_price_ml.evaluation.gate_config import load_quality_gates
from house_price_ml.evaluation.gates import GateResult, evaluate_training_gates
from house_price_ml.evaluation.metrics import compute_metrics, evaluate_by_segment
from house_price_ml.evaluation.segments import price_category
from house_price_ml.jobs.databricks_train import (
    apply_experiment_flags,
    parse_bool,
    training_config_from_job_params,
)
from house_price_ml.jobs.experiment_lab import (
    LAB_MLFLOW_EXPERIMENT,
    configure_lab_mlflow,
    data_quality_summary,
    format_gate_report,
    load_training_frame_from_catalog,
    load_training_frame_sample,
    search_recent_runs,
)
from house_price_ml.models.baseline import BusinessBaseline
from house_price_ml.models.train import _git_commit, train

BUSINESS_BASELINE_LABEL = (
    "Business baseline: median €/m² by region × property type × surface"
)

warnings.filterwarnings("ignore")


dbutils.widgets.dropdown(
    "catalog",
    "house_price_staging",
    ["house_price_staging", "house_price_prod"],
)
dbutils.widgets.dropdown("data_source", "sample", ["delta", "sample"])
dbutils.widgets.dropdown("sample_profile", "demo", ["demo", "engineering"])
dbutils.widgets.text("sample_rows", "")
dbutils.widgets.text(
    "mlflow_experiment",
    "/Shared/house_price_prediction_lab_enterprise",
)
dbutils.widgets.text(
    "eda_config_path",
    "ml/config/eda_lab_enterprise.yaml",
)
dbutils.widgets.dropdown("run_rf_search", "true", ["true", "false"])
dbutils.widgets.dropdown("run_catboost_search", "true", ["true", "false"])
dbutils.widgets.dropdown("run_shap", "false", ["true", "false"])
dbutils.widgets.dropdown("run_full_train_dry_run", "false", ["true", "false"])
dbutils.widgets.dropdown("enable_tuning", "false", ["true", "false"])
dbutils.widgets.dropdown("enable_ablation", "false", ["true", "false"])
dbutils.widgets.dropdown("enable_explainability", "false", ["true", "false"])
dbutils.widgets.text("git_commit", "")


catalog = _widget("catalog", "house_price_staging")
data_source = _widget("data_source", "sample")
sample_profile = _widget("sample_profile", "demo")
sample_rows_raw = _widget("sample_rows", "").strip()
sample_rows = int(sample_rows_raw) if sample_rows_raw else None
mlflow_experiment = _widget(
    "mlflow_experiment",
    "/Shared/house_price_prediction_lab_enterprise",
)
eda_config_path = _widget(
    "eda_config_path",
    "ml/config/eda_lab_enterprise.yaml",
).strip()
run_rf_search = parse_bool(_widget("run_rf_search", "true"))
run_catboost_search = parse_bool(_widget("run_catboost_search", "true"))
run_shap = parse_bool(_widget("run_shap", "false"))
run_full_train_dry_run = parse_bool(
    _widget("run_full_train_dry_run", "false")
)
enable_tuning = parse_bool(_widget("enable_tuning", "false"))
enable_ablation = parse_bool(_widget("enable_ablation", "false"))
enable_explainability = parse_bool(
    _widget("enable_explainability", "false")
)
widget_commit = _widget("git_commit", "").strip()
if widget_commit and widget_commit not in ("unknown", "none", ""):
    os.environ["GIT_COMMIT"] = widget_commit

lab_git_commit = _git_commit(
    widget_commit
    if widget_commit not in ("unknown", "none", "")
    else None
)

training_config = load_training_config()
gates_config = load_quality_gates()
tracking_uri = configure_lab_mlflow(mlflow_experiment)

print("Lane contract: lab-only, register_model=False.")
print(f"Catalog: {catalog}")
print(f"Data source: {data_source}")
print(f"MLflow experiment: {mlflow_experiment}")
print(f"EDA config: {eda_config_path}")
print(f"Git commit: {lab_git_commit}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Configuration loader

# COMMAND ----------

def _load_yaml(path: str) -> dict[str, Any]:
    import yaml

    candidate_paths = [
        Path(path),
        Path.cwd() / path,
        Path("/Workspace") / path.lstrip("/"),
    ]

    for candidate in candidate_paths:
        if candidate.exists():
            payload = yaml.safe_load(candidate.read_text()) or {}
            payload["_resolved_path"] = str(candidate)
            return payload

    raise FileNotFoundError(
        f"Could not locate YAML config: {path}. "
        f"Tried: {[str(p) for p in candidate_paths]}"
    )


LAB_CONFIG = _load_yaml(eda_config_path)

TARGET_COLUMN = LAB_CONFIG.get("target_column", "label_sale_price")
DATE_COLUMN = LAB_CONFIG.get("date_column", "sale_date")
ID_COLUMN = LAB_CONFIG.get("id_column", "listing_id")

TRAIN_FRACTION = float(
    LAB_CONFIG.get("split", {}).get("train_fraction", 0.70)
)
VALIDATION_FRACTION = float(
    LAB_CONFIG.get("split", {}).get("validation_fraction", 0.15)
)
TEST_FRACTION = 1.0 - TRAIN_FRACTION - VALIDATION_FRACTION

if not 0 < TRAIN_FRACTION < 1:
    raise ValueError("train_fraction must be between 0 and 1")
if not 0 < VALIDATION_FRACTION < 1:
    raise ValueError("validation_fraction must be between 0 and 1")
if TEST_FRACTION <= 0:
    raise ValueError(
        "train_fraction + validation_fraction must be less than 1"
    )

SECTIONS = LAB_CONFIG.get("sections", {})
DQ_CONFIG = LAB_CONFIG.get("data_quality", {})
EDA_CONFIG = LAB_CONFIG.get("eda", {})
FEATURE_MATRIX_CONFIG = LAB_CONFIG.get("feature_matrix", {})
MODEL_SELECTION_CONFIG = LAB_CONFIG.get("model_selection", {})
SELECTION_CONFIG = LAB_CONFIG.get("candidate_selection", {})
RESIDUAL_CONFIG = LAB_CONFIG.get("residual_analysis", {})
SHAP_CONFIG = LAB_CONFIG.get("shap", {})

RANDOM_STATE = int(LAB_CONFIG.get("random_state", 42))

print(json.dumps(LAB_CONFIG, indent=2, default=str))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Load source and model-ready data

# COMMAND ----------

source_df: pd.DataFrame
model_df: pd.DataFrame
rejected_df: pd.DataFrame | None = None

if data_source == "delta":
    source_df = spark.table(
        f"{catalog}.silver.listings_clean"
    ).toPandas()

    try:
        rejected_df = spark.table(
            f"{catalog}.silver.listings_rejected"
        ).toPandas()
    except Exception:
        rejected_df = None

    model_df = load_training_frame_from_catalog(spark, catalog)
    print(
        f"Loaded source rows={len(source_df):,}, "
        f"model rows={len(model_df):,}"
    )
else:
    sample_df = load_training_frame_sample(
        profile=sample_profile,
        rows=sample_rows,
    )
    source_df = sample_df.copy()
    model_df = sample_df.copy()
    print(
        f"Loaded in-memory sample rows={len(model_df):,} "
        f"(profile={sample_profile})"
    )

display(source_df.head(5))
display(model_df.head(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Dataset inventory

# COMMAND ----------

def infer_feature_inventory(
    df: pd.DataFrame,
    *,
    target_column: str,
    date_column: str,
    id_column: str,
    excluded_columns: Sequence[str],
) -> pd.DataFrame:
    excluded = set(excluded_columns) | {
        target_column,
        date_column,
        id_column,
    }

    rows: list[dict[str, Any]] = []

    for column in df.columns:
        series = df[column]
        unique_count = int(series.nunique(dropna=False))

        if column == target_column:
            role = "target"
        elif column == date_column:
            role = "date"
        elif column == id_column:
            role = "identifier"
        elif column in excluded:
            role = "excluded"
        elif pd.api.types.is_bool_dtype(series):
            role = "categorical"
        elif pd.api.types.is_numeric_dtype(series):
            role = "numeric"
        elif pd.api.types.is_datetime64_any_dtype(series):
            role = "datetime"
        else:
            role = "categorical"

        rows.append(
            {
                "column": column,
                "role": role,
                "dtype": str(series.dtype),
                "row_count": len(series),
                "missing_count": int(series.isna().sum()),
                "missing_pct": float(series.isna().mean() * 100),
                "unique_count": unique_count,
                "constant_column": unique_count <= 1,
                "high_cardinality": (
                    unique_count > max(20, int(0.50 * len(df)))
                ),
            }
        )

    return pd.DataFrame(rows)


excluded_features = LAB_CONFIG.get(
    "excluded_features",
    ["asking_price", "sale_price_per_sqm"],
)

feature_inventory = infer_feature_inventory(
    model_df,
    target_column=TARGET_COLUMN,
    date_column=DATE_COLUMN,
    id_column=ID_COLUMN,
    excluded_columns=excluded_features,
)

display(
    feature_inventory.sort_values(
        ["role", "missing_pct", "unique_count"],
        ascending=[True, False, False],
    )
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Data-quality funnel and duplicate analysis

# COMMAND ----------

def exact_duplicate_report(
    df: pd.DataFrame,
    duplicate_key: Sequence[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    available_key = [column for column in duplicate_key if column in df.columns]

    if not available_key:
        raise ValueError(
            "None of the configured duplicate-key columns exist in the frame"
        )

    duplicate_mask = df.duplicated(
        subset=available_key,
        keep=False,
    )

    duplicate_groups = (
        df.loc[duplicate_mask]
        .groupby(available_key, dropna=False)
        .size()
        .reset_index(name="duplicate_count")
        .sort_values("duplicate_count", ascending=False)
    )

    summary = pd.DataFrame(
        [
            {
                "metric": "raw_rows",
                "value": len(df),
            },
            {
                "metric": "rows_in_duplicate_groups",
                "value": int(duplicate_mask.sum()),
            },
            {
                "metric": "exact_duplicate_groups",
                "value": len(duplicate_groups),
            },
            {
                "metric": "unique_rows_after_deduplication",
                "value": len(
                    df.drop_duplicates(
                        subset=available_key,
                        keep="first",
                    )
                ),
            },
            {
                "metric": "duplicate_group_row_rate",
                "value": float(duplicate_mask.mean()),
            },
        ]
    )

    return summary, duplicate_groups


duplicate_key = DQ_CONFIG.get(
    "duplicate_key",
    [
        "region",
        "postcode",
        "property_type",
        "energy_label",
        "surface_area",
        "number_of_rooms",
        "number_of_bedrooms",
        "build_year",
        "latitude",
        "longitude",
        "garden",
        TARGET_COLUMN,
        DATE_COLUMN,
    ],
)

duplicate_summary, duplicate_groups = exact_duplicate_report(
    source_df,
    duplicate_key,
)

display(duplicate_summary)
display(duplicate_groups.head(30))

dq_summary = data_quality_summary(source_df, rejected_df)
dq_summary["duplicate_summary"] = dict(
    zip(duplicate_summary["metric"], duplicate_summary["value"])
)

max_duplicate_rate = float(
    DQ_CONFIG.get("max_duplicate_group_row_rate", 0.01)
)
observed_duplicate_rate = float(
    duplicate_summary.loc[
        duplicate_summary["metric"] == "duplicate_group_row_rate",
        "value",
    ].iloc[0]
)

dq_gate_table = pd.DataFrame(
    [
        {
            "gate": "duplicate_group_row_rate",
            "observed": observed_duplicate_rate,
            "threshold": max_duplicate_rate,
            "passed": observed_duplicate_rate <= max_duplicate_rate,
        }
    ]
)

display(dq_gate_table)

if rejected_df is not None and len(rejected_df):
    display(rejected_df.head(20))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Cleaning contract

# COMMAND ----------

def clean_source_frame(
    df: pd.DataFrame,
    *,
    target_column: str,
    date_column: str,
    id_column: str,
    duplicate_key: Sequence[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    result = df.copy()
    funnel: list[dict[str, Any]] = []

    funnel.append(
        {
            "step": "raw",
            "removed": 0,
            "remaining": len(result),
        }
    )

    if date_column in result.columns:
        result[date_column] = pd.to_datetime(
            result[date_column],
            errors="coerce",
        )

    numeric_candidates = [
        column
        for column in result.columns
        if pd.api.types.is_numeric_dtype(result[column])
    ]

    for column in numeric_candidates:
        result[column] = pd.to_numeric(
            result[column],
            errors="coerce",
        )

    required = [
        column
        for column in [
            target_column,
            date_column,
            "surface_area",
            "region",
        ]
        if column in result.columns
    ]

    before = len(result)
    result = result.dropna(subset=required)
    funnel.append(
        {
            "step": "drop_missing_required",
            "removed": before - len(result),
            "remaining": len(result),
        }
    )

    validity_mask = pd.Series(True, index=result.index)

    if target_column in result.columns:
        validity_mask &= result[target_column] > 0

    if "surface_area" in result.columns:
        validity_mask &= result["surface_area"] > 0

    before = len(result)
    result = result.loc[validity_mask].copy()
    funnel.append(
        {
            "step": "drop_invalid_values",
            "removed": before - len(result),
            "remaining": len(result),
        }
    )

    available_duplicate_key = [
        column for column in duplicate_key if column in result.columns
    ]

    before = len(result)
    result = result.drop_duplicates(
        subset=available_duplicate_key,
        keep="first",
    ).copy()
    funnel.append(
        {
            "step": "drop_exact_duplicates",
            "removed": before - len(result),
            "remaining": len(result),
        }
    )

    sort_columns = [
        column
        for column in [date_column, id_column]
        if column in result.columns
    ]

    if sort_columns:
        result = result.sort_values(sort_columns)

    result = result.reset_index(drop=True)

    funnel.append(
        {
            "step": "final",
            "removed": 0,
            "remaining": len(result),
        }
    )

    return result, pd.DataFrame(funnel)


clean_source_df, cleaning_funnel = clean_source_frame(
    source_df,
    target_column=TARGET_COLUMN,
    date_column=DATE_COLUMN,
    id_column=ID_COLUMN,
    duplicate_key=duplicate_key,
)

display(cleaning_funnel)

# Keep the model-ready frame from Gold if available.
# If source and model frames have the same schema, use the cleaned frame.
if set(model_df.columns) == set(source_df.columns):
    model_df = clean_source_df.copy()
else:
    model_df = model_df.copy()
    if DATE_COLUMN in model_df.columns:
        model_df[DATE_COLUMN] = pd.to_datetime(
            model_df[DATE_COLUMN],
            errors="coerce",
        )
    if ID_COLUMN in model_df.columns:
        model_df = model_df.drop_duplicates(subset=[ID_COLUMN]).copy()
    model_df = model_df.sort_values(
        [column for column in [DATE_COLUMN, ID_COLUMN] if column in model_df.columns]
    ).reset_index(drop=True)

print(f"Clean source rows: {len(clean_source_df):,}")
print(f"Model rows: {len(model_df):,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Dynamic univariate EDA

# COMMAND ----------

def univariate_numeric_profile(
    df: pd.DataFrame,
    columns: Sequence[str],
) -> pd.DataFrame:
    available = [column for column in columns if column in df.columns]
    if not available:
        return pd.DataFrame()

    profile = (
        df[available]
        .describe(
            percentiles=[0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]
        )
        .T
    )

    profile["missing_count"] = df[available].isna().sum()
    profile["missing_pct"] = df[available].isna().mean() * 100
    profile["skew"] = df[available].skew(numeric_only=True)

    def iqr_outlier_rate(series: pd.Series) -> float:
        values = series.dropna()
        if values.empty:
            return float("nan")

        q1 = values.quantile(0.25)
        q3 = values.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            return 0.0

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        return float(((values < lower) | (values > upper)).mean())

    profile["iqr_outlier_rate"] = [
        iqr_outlier_rate(df[column]) for column in available
    ]

    return profile


numeric_columns = (
    feature_inventory.loc[
        feature_inventory["role"] == "numeric",
        "column",
    ]
    .tolist()
)

categorical_columns = (
    feature_inventory.loc[
        feature_inventory["role"] == "categorical",
        "column",
    ]
    .tolist()
)

if SECTIONS.get("univariate", True):
    numeric_profile = univariate_numeric_profile(
        model_df,
        numeric_columns,
    )
    display(numeric_profile)

    max_plots = int(EDA_CONFIG.get("max_numeric_plots", 30))
    histogram_bins = int(EDA_CONFIG.get("histogram_bins", 30))

    for column in numeric_columns[:max_plots]:
        values = model_df[column].dropna()
        if values.empty:
            continue

        plt.figure(figsize=(9, 4))
        plt.hist(values, bins=histogram_bins)
        plt.axvline(values.median(), linestyle="--")
        plt.xlabel(column)
        plt.ylabel("Count")
        plt.title(f"Distribution — {column}")
        plt.tight_layout()
        plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Dynamic categorical EDA

# COMMAND ----------

def categorical_profile(
    df: pd.DataFrame,
    columns: Sequence[str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for column in columns:
        if column not in df.columns:
            continue

        values = (
            df[column]
            .astype("string")
            .fillna("__MISSING__")
            .value_counts(dropna=False)
        )

        rows.append(
            {
                "column": column,
                "unique_count": len(values),
                "missing_count": int(df[column].isna().sum()),
                "missing_pct": float(df[column].isna().mean() * 100),
                "most_common_value": (
                    str(values.index[0]) if len(values) else None
                ),
                "most_common_count": (
                    int(values.iloc[0]) if len(values) else 0
                ),
                "most_common_pct": (
                    float(values.iloc[0] / len(df) * 100)
                    if len(values) and len(df)
                    else float("nan")
                ),
            }
        )

    return pd.DataFrame(rows)


if SECTIONS.get("categorical_eda", True):
    categorical_summary = categorical_profile(
        model_df,
        categorical_columns,
    )
    display(categorical_summary)

    max_categories = int(
        EDA_CONFIG.get("max_categories_per_plot", 15)
    )
    max_plots = int(
        EDA_CONFIG.get("max_categorical_plots", 30)
    )

    for column in categorical_columns[:max_plots]:
        counts = (
            model_df[column]
            .astype("string")
            .fillna("__MISSING__")
            .value_counts()
            .head(max_categories)
            .sort_values()
        )

        if counts.empty:
            continue

        plt.figure(
            figsize=(9, max(4, len(counts) * 0.35))
        )
        plt.barh(counts.index.astype(str), counts.values)
        plt.xlabel("Count")
        plt.ylabel(column)
        plt.title(f"Most common categories — {column}")
        plt.tight_layout()
        plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Dynamic bivariate EDA and correlation

# COMMAND ----------

def numeric_target_report(
    df: pd.DataFrame,
    *,
    target_column: str,
    numeric_features: Sequence[str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for feature in numeric_features:
        if feature not in df.columns:
            continue

        pair = df[[feature, target_column]].dropna()
        if len(pair) < 3:
            continue

        correlation = pair[feature].corr(pair[target_column])

        rows.append(
            {
                "feature": feature,
                "count": len(pair),
                "pearson_correlation": correlation,
                "correlation_squared": (
                    correlation**2 if pd.notna(correlation) else np.nan
                ),
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values(
            "pearson_correlation",
            key=lambda values: values.abs(),
            ascending=False,
        )
        .reset_index(drop=True)
    )


def categorical_target_report(
    df: pd.DataFrame,
    *,
    target_column: str,
    categorical_features: Sequence[str],
    minimum_group_size: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    overall_mean = float(df[target_column].mean())

    for feature in categorical_features:
        if feature not in df.columns:
            continue

        grouped = (
            df.assign(
                **{
                    feature: (
                        df[feature]
                        .astype("string")
                        .fillna("__MISSING__")
                    )
                }
            )
            .groupby(feature)[target_column]
            .agg(["count", "mean", "median", "std"])
            .reset_index()
        )

        grouped = grouped[grouped["count"] >= minimum_group_size]

        for _, row in grouped.iterrows():
            rows.append(
                {
                    "feature": feature,
                    "category": row[feature],
                    "count": int(row["count"]),
                    "mean_target": float(row["mean"]),
                    "median_target": float(row["median"]),
                    "std_target": float(row["std"])
                    if pd.notna(row["std"])
                    else np.nan,
                    "mean_delta_vs_overall": (
                        float(row["mean"]) - overall_mean
                    ),
                }
            )

    return pd.DataFrame(rows)


if SECTIONS.get("bivariate", True):
    numeric_bivariate = numeric_target_report(
        model_df,
        target_column=TARGET_COLUMN,
        numeric_features=numeric_columns,
    )
    display(numeric_bivariate)

    categorical_bivariate = categorical_target_report(
        model_df,
        target_column=TARGET_COLUMN,
        categorical_features=categorical_columns,
        minimum_group_size=int(
            EDA_CONFIG.get("minimum_category_group_size", 5)
        ),
    )
    display(categorical_bivariate)

    correlation_columns = [
        column
        for column in numeric_columns + [TARGET_COLUMN]
        if column in model_df.columns
    ]

    correlation_matrix = model_df[
        correlation_columns
    ].corr(numeric_only=True)

    display(correlation_matrix)

    plt.figure(
        figsize=(
            max(8, len(correlation_columns) * 0.7),
            max(7, len(correlation_columns) * 0.7),
        )
    )
    image = plt.imshow(
        correlation_matrix,
        aspect="auto",
        vmin=-1,
        vmax=1,
    )
    plt.colorbar(image, label="Pearson correlation")
    plt.xticks(
        range(len(correlation_columns)),
        correlation_columns,
        rotation=90,
    )
    plt.yticks(
        range(len(correlation_columns)),
        correlation_columns,
    )
    plt.title("Numeric correlation matrix")
    plt.tight_layout()
    plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Time and drift analysis

# COMMAND ----------

def monthly_target_drift(
    df: pd.DataFrame,
    *,
    date_column: str,
    target_column: str,
) -> pd.DataFrame:
    working = df[[date_column, target_column]].dropna().copy()
    working[date_column] = pd.to_datetime(
        working[date_column],
        errors="coerce",
    )

    return (
        working.set_index(date_column)
        .resample("ME")[target_column]
        .agg(["count", "mean", "median"])
        .reset_index()
    )


if SECTIONS.get("time_drift", True):
    drift_table = monthly_target_drift(
        model_df,
        date_column=DATE_COLUMN,
        target_column=TARGET_COLUMN,
    )
    display(drift_table)

    plt.figure(figsize=(11, 5))
    plt.plot(
        drift_table[DATE_COLUMN],
        drift_table["median"],
    )
    plt.xlabel("Month")
    plt.ylabel(f"Median {TARGET_COLUMN}")
    plt.title("Target drift over time")
    plt.tight_layout()
    plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 11. Business hypotheses
# MAGIC
# MAGIC Hypotheses are explicit, but broad feature discovery is still allowed.

# COMMAND ----------

def evaluate_configured_hypotheses(
    df: pd.DataFrame,
    hypotheses: Sequence[dict[str, Any]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for hypothesis in hypotheses:
        name = hypothesis.get("name", "unnamed")
        feature = hypothesis.get("feature")
        target = hypothesis.get("target", TARGET_COLUMN)
        kind = hypothesis.get("kind", "numeric_correlation")

        try:
            if kind == "numeric_correlation":
                pair = df[[feature, target]].dropna()
                value = pair[feature].corr(pair[target])
                passed = abs(value) >= float(
                    hypothesis.get("minimum_absolute_correlation", 0.0)
                )
                details = {
                    "count": len(pair),
                    "correlation": value,
                }

            elif kind == "categorical_spread":
                grouped = (
                    df.groupby(feature)[target]
                    .agg(["count", "mean"])
                    .query(
                        f"count >= {int(hypothesis.get('minimum_group_size', 5))}"
                    )
                )
                spread = (
                    float(grouped["mean"].max() - grouped["mean"].min())
                    if len(grouped)
                    else np.nan
                )
                passed = spread >= float(
                    hypothesis.get("minimum_mean_spread", 0.0)
                )
                details = {
                    "groups": len(grouped),
                    "mean_spread": spread,
                }

            else:
                raise ValueError(f"Unsupported hypothesis kind: {kind}")

            rows.append(
                {
                    "name": name,
                    "kind": kind,
                    "feature": feature,
                    "passed": bool(passed),
                    "details": json.dumps(details, default=str),
                }
            )

        except Exception as exc:
            rows.append(
                {
                    "name": name,
                    "kind": kind,
                    "feature": feature,
                    "passed": False,
                    "details": f"ERROR: {exc}",
                }
            )

    return pd.DataFrame(rows)


hypothesis_results = evaluate_configured_hypotheses(
    model_df,
    LAB_CONFIG.get("business_hypotheses", []),
)
display(hypothesis_results)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 12. Chronological train / validation / test split
# MAGIC
# MAGIC Date boundaries are used so one sale date cannot appear in multiple sets.

# COMMAND ----------

@dataclass(frozen=True)
class ChronologicalSplit:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    train_end_date: pd.Timestamp
    validation_end_date: pd.Timestamp


def chronological_split_by_date(
    df: pd.DataFrame,
    *,
    date_column: str,
    train_fraction: float,
    validation_fraction: float,
) -> ChronologicalSplit:
    working = df.copy()
    working[date_column] = pd.to_datetime(
        working[date_column],
        errors="coerce",
    )
    working = (
        working.dropna(subset=[date_column])
        .sort_values(
            [
                column
                for column in [date_column, ID_COLUMN]
                if column in working.columns
            ]
        )
        .reset_index(drop=True)
    )

    unique_dates = pd.Series(
        working[date_column].dropna().sort_values().unique()
    )

    if len(unique_dates) < 3:
        raise ValueError(
            "At least three unique dates are required for "
            "train/validation/test splitting"
        )

    train_date_index = max(
        0,
        min(
            len(unique_dates) - 3,
            int(len(unique_dates) * train_fraction) - 1,
        ),
    )
    validation_date_index = max(
        train_date_index + 1,
        min(
            len(unique_dates) - 2,
            int(
                len(unique_dates)
                * (train_fraction + validation_fraction)
            )
            - 1,
        ),
    )

    train_end_date = pd.Timestamp(
        unique_dates.iloc[train_date_index]
    )
    validation_end_date = pd.Timestamp(
        unique_dates.iloc[validation_date_index]
    )

    train = working[
        working[date_column] <= train_end_date
    ].copy()
    validation = working[
        (working[date_column] > train_end_date)
        & (working[date_column] <= validation_end_date)
    ].copy()
    test = working[
        working[date_column] > validation_end_date
    ].copy()

    if min(len(train), len(validation), len(test)) == 0:
        raise ValueError(
            "Chronological split produced an empty partition"
        )

    return ChronologicalSplit(
        train=train,
        validation=validation,
        test=test,
        train_end_date=train_end_date,
        validation_end_date=validation_end_date,
    )


split = chronological_split_by_date(
    model_df,
    date_column=DATE_COLUMN,
    train_fraction=TRAIN_FRACTION,
    validation_fraction=VALIDATION_FRACTION,
)

train_df = split.train
validation_df = split.validation
test_df = split.test

split_summary = pd.DataFrame(
    [
        {
            "partition": "train",
            "rows": len(train_df),
            "start": train_df[DATE_COLUMN].min(),
            "end": train_df[DATE_COLUMN].max(),
        },
        {
            "partition": "validation",
            "rows": len(validation_df),
            "start": validation_df[DATE_COLUMN].min(),
            "end": validation_df[DATE_COLUMN].max(),
        },
        {
            "partition": "test",
            "rows": len(test_df),
            "start": test_df[DATE_COLUMN].min(),
            "end": test_df[DATE_COLUMN].max(),
        },
    ]
)

display(split_summary)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 13. Metrics and business baseline

# COMMAND ----------

def regression_metrics(
    actual: Sequence[float],
    predicted: Sequence[float],
) -> dict[str, float]:
    actual_array = np.asarray(actual, dtype=float)
    predicted_array = np.asarray(predicted, dtype=float)

    non_zero = actual_array != 0

    return {
        "mae": float(
            mean_absolute_error(actual_array, predicted_array)
        ),
        "median_ae": float(
            np.median(np.abs(actual_array - predicted_array))
        ),
        "rmse": float(
            np.sqrt(
                mean_squared_error(
                    actual_array,
                    predicted_array,
                )
            )
        ),
        "r2": float(
            r2_score(actual_array, predicted_array)
        ),
        "mape_pct": float(
            np.mean(
                np.abs(
                    (
                        actual_array[non_zero]
                        - predicted_array[non_zero]
                    )
                    / actual_array[non_zero]
                )
            )
            * 100
        )
        if non_zero.any()
        else float("nan"),
    }


validation_baseline = BusinessBaseline().fit(train_df)
validation_baseline_predictions = validation_baseline.predict(
    validation_df
)

validation_baseline_metrics = regression_metrics(
    validation_df[TARGET_COLUMN],
    validation_baseline_predictions,
)

display(
    pd.DataFrame(
        [
            {
                "model": BUSINESS_BASELINE_LABEL,
                **validation_baseline_metrics,
            }
        ]
    )
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 14. Single-feature predictive screen
# MAGIC
# MAGIC Each candidate feature is fitted alone on train and evaluated on validation.

# COMMAND ----------

def _build_sklearn_pipeline(
    *,
    numeric_features: Sequence[str],
    categorical_features: Sequence[str],
    estimator: Any,
) -> Pipeline:
    transformers: list[tuple[str, Any, Sequence[str]]] = []

    if numeric_features:
        transformers.append(
            (
                "numeric",
                Pipeline(
                    steps=[
                        (
                            "imputer",
                            SimpleImputer(strategy="median"),
                        )
                    ]
                ),
                list(numeric_features),
            )
        )

    if categorical_features:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    steps=[
                        (
                            "imputer",
                            SimpleImputer(
                                strategy="most_frequent"
                            ),
                        ),
                        (
                            "onehot",
                            OneHotEncoder(
                                handle_unknown="ignore",
                                min_frequency=3,
                            ),
                        ),
                    ]
                ),
                list(categorical_features),
            )
        )

    return Pipeline(
        steps=[
            (
                "preprocessor",
                ColumnTransformer(
                    transformers=transformers,
                    remainder="drop",
                ),
            ),
            ("model", estimator),
        ]
    )


def run_single_feature_screen(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    *,
    target_column: str,
    inventory: pd.DataFrame,
    baseline_mae: float,
    random_state: int,
) -> pd.DataFrame:
    candidate_inventory = inventory[
        inventory["role"].isin(["numeric", "categorical"])
    ].copy()

    rows: list[dict[str, Any]] = []

    for _, item in candidate_inventory.iterrows():
        feature = item["column"]
        feature_type = item["role"]

        if feature not in train_df.columns:
            continue

        estimator = RandomForestRegressor(
            n_estimators=200,
            min_samples_leaf=5,
            random_state=random_state,
            n_jobs=-1,
        )

        pipeline = _build_sklearn_pipeline(
            numeric_features=(
                [feature] if feature_type == "numeric" else []
            ),
            categorical_features=(
                [feature] if feature_type == "categorical" else []
            ),
            estimator=estimator,
        )

        try:
            start = time.perf_counter()
            pipeline.fit(
                train_df[[feature]],
                train_df[target_column],
            )
            predictions = pipeline.predict(
                validation_df[[feature]]
            )
            elapsed = time.perf_counter() - start
            metrics = regression_metrics(
                validation_df[target_column],
                predictions,
            )

            rows.append(
                {
                    "feature": feature,
                    "feature_type": feature_type,
                    **metrics,
                    "mae_improvement_vs_baseline": (
                        baseline_mae - metrics["mae"]
                    ),
                    "mae_improvement_vs_baseline_pct": (
                        (baseline_mae - metrics["mae"])
                        / baseline_mae
                        * 100
                    ),
                    "training_seconds": elapsed,
                    "status": "ok",
                }
            )

        except Exception as exc:
            rows.append(
                {
                    "feature": feature,
                    "feature_type": feature_type,
                    "status": f"error: {exc}",
                }
            )

    result = pd.DataFrame(rows)

    if "mae" in result.columns:
        result = result.sort_values(
            ["mae", "r2"],
            ascending=[True, False],
        ).reset_index(drop=True)
        result.insert(
            0,
            "rank",
            np.arange(1, len(result) + 1),
        )

    return result


single_feature_results = run_single_feature_screen(
    train_df,
    validation_df,
    target_column=TARGET_COLUMN,
    inventory=feature_inventory,
    baseline_mae=validation_baseline_metrics["mae"],
    random_state=RANDOM_STATE,
)

display(single_feature_results)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 15. Feature groups and generated combination matrix

# COMMAND ----------

def validate_feature_groups(
    feature_groups: dict[str, list[str]],
    df: pd.DataFrame,
) -> tuple[dict[str, list[str]], pd.DataFrame]:
    valid_groups: dict[str, list[str]] = {}
    rows: list[dict[str, Any]] = []

    for group_name, features in feature_groups.items():
        available = [
            feature for feature in features if feature in df.columns
        ]
        missing = [
            feature for feature in features if feature not in df.columns
        ]

        rows.append(
            {
                "group": group_name,
                "configured_feature_count": len(features),
                "available_feature_count": len(available),
                "available_features": available,
                "missing_features": missing,
                "enabled": bool(available),
            }
        )

        if available:
            valid_groups[group_name] = available

    return valid_groups, pd.DataFrame(rows)


configured_feature_groups = FEATURE_MATRIX_CONFIG.get(
    "groups",
    {},
)

feature_groups, feature_group_inventory = (
    validate_feature_groups(
        configured_feature_groups,
        model_df,
    )
)

display(feature_group_inventory)

MAX_GROUP_DEPTH = int(
    FEATURE_MATRIX_CONFIG.get("max_depth", 4)
)
MAX_EXPERIMENTS = int(
    FEATURE_MATRIX_CONFIG.get("max_experiments", 500)
)


def generated_group_combinations(
    feature_groups: dict[str, list[str]],
    *,
    max_depth: int,
    max_experiments: int,
) -> list[tuple[str, ...]]:
    group_names = list(feature_groups)

    combinations: list[tuple[str, ...]] = []

    for depth in range(
        1,
        min(max_depth, len(group_names)) + 1,
    ):
        combinations.extend(
            itertools.combinations(group_names, depth)
        )

    if len(combinations) > max_experiments:
        raise ValueError(
            f"Generated {len(combinations)} experiments, "
            f"exceeding max_experiments={max_experiments}. "
            "Reduce max_depth or number of feature groups."
        )

    return combinations


feature_group_combinations = generated_group_combinations(
    feature_groups,
    max_depth=MAX_GROUP_DEPTH,
    max_experiments=MAX_EXPERIMENTS,
)

print(
    f"Generated feature-group experiments: "
    f"{len(feature_group_combinations):,}"
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 16. Feature search reference model: Random Forest

# COMMAND ----------

def feature_list_from_groups(
    combination: Sequence[str],
    feature_groups: dict[str, list[str]],
) -> list[str]:
    features: list[str] = []

    for group in combination:
        features.extend(feature_groups[group])

    return list(dict.fromkeys(features))


def split_feature_types(
    features: Sequence[str],
    inventory: pd.DataFrame,
) -> tuple[list[str], list[str]]:
    role_by_column = dict(
        zip(inventory["column"], inventory["role"])
    )

    numeric = [
        feature
        for feature in features
        if role_by_column.get(feature) == "numeric"
    ]
    categorical = [
        feature
        for feature in features
        if role_by_column.get(feature) == "categorical"
    ]

    return numeric, categorical


def run_rf_feature_matrix(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    *,
    combinations: Sequence[Sequence[str]],
    feature_groups: dict[str, list[str]],
    inventory: pd.DataFrame,
    target_column: str,
    baseline_mae: float,
    config: dict[str, Any],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    n_estimators = int(config.get("n_estimators", 200))
    min_samples_leaf = int(
        config.get("min_samples_leaf", 3)
    )
    max_features = config.get("max_features", "sqrt")

    for index, combination in enumerate(combinations, start=1):
        features = feature_list_from_groups(
            combination,
            feature_groups,
        )
        numeric, categorical = split_feature_types(
            features,
            inventory,
        )

        pipeline = _build_sklearn_pipeline(
            numeric_features=numeric,
            categorical_features=categorical,
            estimator=RandomForestRegressor(
                n_estimators=n_estimators,
                min_samples_leaf=min_samples_leaf,
                max_features=max_features,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
        )

        start = time.perf_counter()
        pipeline.fit(
            train_df[features],
            train_df[target_column],
        )
        predictions = pipeline.predict(
            validation_df[features]
        )
        elapsed = time.perf_counter() - start
        metrics = regression_metrics(
            validation_df[target_column],
            predictions,
        )

        rows.append(
            {
                "model_family": "random_forest",
                "experiment": " + ".join(combination),
                "groups": tuple(combination),
                "depth": len(combination),
                "number_of_raw_features": len(features),
                "feature_list": features,
                **metrics,
                "mae_improvement_vs_baseline": (
                    baseline_mae - metrics["mae"]
                ),
                "mae_improvement_vs_baseline_pct": (
                    (baseline_mae - metrics["mae"])
                    / baseline_mae
                    * 100
                ),
                "beats_baseline": metrics["mae"] < baseline_mae,
                "training_seconds": elapsed,
            }
        )

        if index % 50 == 0 or index == len(combinations):
            print(
                f"Random Forest feature search: "
                f"{index}/{len(combinations)}"
            )

    return (
        pd.DataFrame(rows)
        .sort_values(["mae", "depth"])
        .reset_index(drop=True)
    )


rf_feature_matrix = pd.DataFrame()

if run_rf_search and SECTIONS.get("feature_matrix", True):
    rf_feature_matrix = run_rf_feature_matrix(
        train_df,
        validation_df,
        combinations=feature_group_combinations,
        feature_groups=feature_groups,
        inventory=feature_inventory,
        target_column=TARGET_COLUMN,
        baseline_mae=validation_baseline_metrics["mae"],
        config=FEATURE_MATRIX_CONFIG.get(
            "random_forest",
            {},
        ),
    )
    rf_feature_matrix.insert(
        0,
        "rank",
        np.arange(1, len(rf_feature_matrix) + 1),
    )
    display(rf_feature_matrix.head(50))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 17. Feature-set stability analysis

# COMMAND ----------

def feature_matrix_stability(
    results: pd.DataFrame,
    *,
    feature_groups: dict[str, list[str]],
    top_n: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if results.empty:
        return pd.DataFrame(), pd.DataFrame()

    best_by_depth = (
        results.sort_values("mae")
        .groupby("depth", as_index=False)
        .first()
        .sort_values("depth")
    )

    top = results.nsmallest(
        min(top_n, len(results)),
        "mae",
    )

    group_frequency_rows = []

    for group in feature_groups:
        share = float(
            top["groups"].apply(
                lambda groups: group in groups
            ).mean()
        )

        group_frequency_rows.append(
            {
                "group": group,
                "top_n": len(top),
                "share_in_top_n": share,
                "count_in_top_n": int(
                    round(share * len(top))
                ),
            }
        )

    group_frequency = (
        pd.DataFrame(group_frequency_rows)
        .sort_values(
            "share_in_top_n",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    return best_by_depth, group_frequency


best_by_depth, top_group_frequency = (
    feature_matrix_stability(
        rf_feature_matrix,
        feature_groups=feature_groups,
        top_n=int(
            FEATURE_MATRIX_CONFIG.get(
                "stability_top_n",
                20,
            )
        ),
    )
)

display(best_by_depth)
display(top_group_frequency)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 18. Candidate feature-set selection
# MAGIC
# MAGIC Select the simplest feature set within a configurable tolerance of the best validation MAE.

# COMMAND ----------

def select_candidate_feature_sets(
    results: pd.DataFrame,
    *,
    relative_mae_tolerance: float,
    top_k_for_model_family_search: int,
) -> pd.DataFrame:
    if results.empty:
        raise ValueError("Feature-matrix results are empty")

    best_mae = float(results["mae"].min())

    near_best = results[
        results["mae"]
        <= best_mae * (1.0 + relative_mae_tolerance)
    ].copy()

    selected_pool = (
        near_best.sort_values(
            [
                "number_of_raw_features",
                "depth",
                "mae",
            ],
            ascending=[True, True, True],
        )
        .head(top_k_for_model_family_search)
        .copy()
    )

    selected_pool["selection_threshold_mae"] = (
        best_mae * (1.0 + relative_mae_tolerance)
    )

    return selected_pool


candidate_feature_sets = select_candidate_feature_sets(
    rf_feature_matrix,
    relative_mae_tolerance=float(
        SELECTION_CONFIG.get(
            "relative_mae_tolerance",
            0.02,
        )
    ),
    top_k_for_model_family_search=int(
        SELECTION_CONFIG.get(
            "top_k_feature_sets",
            3,
        )
    ),
)

display(candidate_feature_sets)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 19. Model-family comparison on selected feature sets

# COMMAND ----------

def prepare_catboost_frame(
    df: pd.DataFrame,
    features: Sequence[str],
    categorical_features: Sequence[str],
) -> pd.DataFrame:
    result = df[list(features)].copy()

    categorical_set = set(categorical_features)

    for feature in features:
        if feature in categorical_set:
            result[feature] = (
                result[feature]
                .astype("string")
                .fillna("__MISSING__")
                .astype(str)
            )
        else:
            result[feature] = pd.to_numeric(
                result[feature],
                errors="coerce",
            )

    return result


def run_model_family_comparison(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    *,
    feature_sets: pd.DataFrame,
    inventory: pd.DataFrame,
    target_column: str,
    baseline_mae: float,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    fitted_models: dict[str, Any] = {}

    compare_random_forest = bool(
        config.get("compare_random_forest", True)
    )
    compare_catboost = bool(
        config.get("compare_catboost", True)
    )

    for candidate_index, candidate in feature_sets.iterrows():
        features = list(candidate["feature_list"])
        numeric, categorical = split_feature_types(
            features,
            inventory,
        )
        feature_set_name = candidate["experiment"]

        if compare_random_forest:
            rf_config = config.get("random_forest", {})
            model_key = (
                f"random_forest::{candidate_index}::{feature_set_name}"
            )

            rf_model = _build_sklearn_pipeline(
                numeric_features=numeric,
                categorical_features=categorical,
                estimator=RandomForestRegressor(
                    n_estimators=int(
                        rf_config.get("n_estimators", 500)
                    ),
                    min_samples_leaf=int(
                        rf_config.get("min_samples_leaf", 3)
                    ),
                    max_features=rf_config.get(
                        "max_features",
                        "sqrt",
                    ),
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            )

            start = time.perf_counter()
            rf_model.fit(
                train_df[features],
                train_df[target_column],
            )
            predictions = rf_model.predict(
                validation_df[features]
            )
            elapsed = time.perf_counter() - start
            metrics = regression_metrics(
                validation_df[target_column],
                predictions,
            )

            rows.append(
                {
                    "model_key": model_key,
                    "model_family": "random_forest",
                    "feature_set": feature_set_name,
                    "feature_list": features,
                    "categorical_features": categorical,
                    "best_iteration": np.nan,
                    **metrics,
                    "mae_improvement_vs_baseline": (
                        baseline_mae - metrics["mae"]
                    ),
                    "training_seconds": elapsed,
                }
            )

            fitted_models[model_key] = rf_model

        if (
            compare_catboost
            and run_catboost_search
            and CatBoostRegressor is not None
        ):
            cat_config = config.get("catboost", {})
            model_key = (
                f"catboost::{candidate_index}::{feature_set_name}"
            )

            X_train = prepare_catboost_frame(
                train_df,
                features,
                categorical,
            )
            X_validation = prepare_catboost_frame(
                validation_df,
                features,
                categorical,
            )

            cat_model = CatBoostRegressor(
                iterations=int(
                    cat_config.get("iterations", 1000)
                ),
                learning_rate=float(
                    cat_config.get("learning_rate", 0.05)
                ),
                depth=int(cat_config.get("depth", 6)),
                loss_function=cat_config.get(
                    "loss_function",
                    "MAE",
                ),
                eval_metric=cat_config.get(
                    "eval_metric",
                    "MAE",
                ),
                cat_features=categorical,
                random_seed=RANDOM_STATE,
                thread_count=-1,
                l2_leaf_reg=float(
                    cat_config.get("l2_leaf_reg", 5.0)
                ),
                random_strength=float(
                    cat_config.get("random_strength", 1.0)
                ),
                verbose=False,
                allow_writing_files=False,
                use_best_model=True,
            )

            start = time.perf_counter()
            cat_model.fit(
                X_train,
                train_df[target_column],
                eval_set=(
                    X_validation,
                    validation_df[target_column],
                ),
                early_stopping_rounds=int(
                    cat_config.get(
                        "early_stopping_rounds",
                        75,
                    )
                ),
                verbose=False,
            )
            predictions = cat_model.predict(X_validation)
            elapsed = time.perf_counter() - start
            metrics = regression_metrics(
                validation_df[target_column],
                predictions,
            )

            best_iteration = cat_model.get_best_iteration()
            best_iteration = (
                int(best_iteration + 1)
                if best_iteration is not None
                and best_iteration >= 0
                else int(cat_config.get("iterations", 1000))
            )

            rows.append(
                {
                    "model_key": model_key,
                    "model_family": "catboost",
                    "feature_set": feature_set_name,
                    "feature_list": features,
                    "categorical_features": categorical,
                    "best_iteration": best_iteration,
                    **metrics,
                    "mae_improvement_vs_baseline": (
                        baseline_mae - metrics["mae"]
                    ),
                    "training_seconds": elapsed,
                }
            )

            fitted_models[model_key] = cat_model

    results = (
        pd.DataFrame(rows)
        .sort_values(["mae", "model_family"])
        .reset_index(drop=True)
    )
    results.insert(
        0,
        "rank",
        np.arange(1, len(results) + 1),
    )

    return results, fitted_models


model_family_results, validation_models = (
    run_model_family_comparison(
        train_df,
        validation_df,
        feature_sets=candidate_feature_sets,
        inventory=feature_inventory,
        target_column=TARGET_COLUMN,
        baseline_mae=validation_baseline_metrics["mae"],
        config=MODEL_SELECTION_CONFIG,
    )
)

display(model_family_results)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 20. Final candidate selection

# COMMAND ----------

def select_final_candidate(
    results: pd.DataFrame,
    *,
    relative_mae_tolerance: float,
) -> pd.Series:
    if results.empty:
        raise ValueError("Model-family results are empty")

    best_mae = float(results["mae"].min())
    near_best = results[
        results["mae"]
        <= best_mae * (1.0 + relative_mae_tolerance)
    ].copy()

    near_best["feature_count"] = near_best[
        "feature_list"
    ].apply(len)

    return (
        near_best.sort_values(
            [
                "feature_count",
                "model_family",
                "mae",
            ],
            ascending=[True, True, True],
        )
        .iloc[0]
    )


selected_candidate = select_final_candidate(
    model_family_results,
    relative_mae_tolerance=float(
        SELECTION_CONFIG.get(
            "final_model_relative_mae_tolerance",
            0.01,
        )
    ),
)

display(selected_candidate.to_frame("value"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 21. Retrain on train + validation and evaluate test once

# COMMAND ----------

train_validation_df = pd.concat(
    [train_df, validation_df],
    axis=0,
).sort_values(DATE_COLUMN)

selected_features = list(
    selected_candidate["feature_list"]
)
selected_model_family = selected_candidate[
    "model_family"
]
selected_categorical_features = list(
    selected_candidate["categorical_features"]
)

if selected_model_family == "random_forest":
    rf_config = MODEL_SELECTION_CONFIG.get(
        "random_forest",
        {},
    )

    final_model = _build_sklearn_pipeline(
        numeric_features=[
            feature
            for feature in selected_features
            if feature not in selected_categorical_features
        ],
        categorical_features=selected_categorical_features,
        estimator=RandomForestRegressor(
            n_estimators=int(
                rf_config.get("final_n_estimators", 800)
            ),
            min_samples_leaf=int(
                rf_config.get("min_samples_leaf", 3)
            ),
            max_features=rf_config.get(
                "max_features",
                "sqrt",
            ),
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    )

    final_model.fit(
        train_validation_df[selected_features],
        train_validation_df[TARGET_COLUMN],
    )
    test_predictions = final_model.predict(
        test_df[selected_features]
    )

elif selected_model_family == "catboost":
    if CatBoostRegressor is None:
        raise RuntimeError(
            "CatBoost selected but catboost is unavailable"
        )

    cat_config = MODEL_SELECTION_CONFIG.get(
        "catboost",
        {},
    )
    final_iterations = max(
        int(selected_candidate["best_iteration"]),
        int(cat_config.get("minimum_final_iterations", 50)),
    )

    X_train_validation = prepare_catboost_frame(
        train_validation_df,
        selected_features,
        selected_categorical_features,
    )
    X_test = prepare_catboost_frame(
        test_df,
        selected_features,
        selected_categorical_features,
    )

    final_model = CatBoostRegressor(
        iterations=final_iterations,
        learning_rate=float(
            cat_config.get("learning_rate", 0.05)
        ),
        depth=int(cat_config.get("depth", 6)),
        loss_function=cat_config.get(
            "loss_function",
            "MAE",
        ),
        eval_metric=cat_config.get(
            "eval_metric",
            "MAE",
        ),
        cat_features=selected_categorical_features,
        random_seed=RANDOM_STATE,
        thread_count=-1,
        l2_leaf_reg=float(
            cat_config.get("l2_leaf_reg", 5.0)
        ),
        random_strength=float(
            cat_config.get("random_strength", 1.0)
        ),
        verbose=False,
        allow_writing_files=False,
    )

    final_model.fit(
        X_train_validation,
        train_validation_df[TARGET_COLUMN],
        verbose=False,
    )
    test_predictions = final_model.predict(X_test)

else:
    raise ValueError(
        f"Unsupported selected model family: "
        f"{selected_model_family}"
    )


test_model_metrics = regression_metrics(
    test_df[TARGET_COLUMN],
    test_predictions,
)

test_baseline = BusinessBaseline().fit(train_validation_df)
test_baseline_predictions = test_baseline.predict(test_df)

test_baseline_metrics = regression_metrics(
    test_df[TARGET_COLUMN],
    test_baseline_predictions,
)

final_test_comparison = pd.DataFrame(
    [
        {
            "model": BUSINESS_BASELINE_LABEL,
            **test_baseline_metrics,
        },
        {
            "model": (
                f"{selected_model_family}: "
                f"{selected_candidate['feature_set']}"
            ),
            **test_model_metrics,
        },
    ]
)

display(final_test_comparison)

test_mae_improvement = (
    test_baseline_metrics["mae"]
    - test_model_metrics["mae"]
)

test_mae_improvement_pct = (
    test_mae_improvement
    / test_baseline_metrics["mae"]
    * 100
)

print(
    "Final test MAE improvement versus baseline: "
    f"{test_mae_improvement:,.2f} "
    f"({test_mae_improvement_pct:.2f}%)"
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 22. Segment evaluation and residual analysis

# COMMAND ----------

residual_columns = [
    column
    for column in [
        ID_COLUMN,
        DATE_COLUMN,
        "region",
        "postcode",
        "property_type",
        "energy_label",
        "surface_area",
        "number_of_rooms",
        "number_of_bedrooms",
        "build_year",
        "garden",
        TARGET_COLUMN,
    ]
    if column in test_df.columns
]

residual_results = test_df[residual_columns].copy()
residual_results["predicted_price"] = test_predictions
residual_results["residual"] = (
    residual_results[TARGET_COLUMN]
    - residual_results["predicted_price"]
)
residual_results["absolute_error"] = (
    residual_results["residual"].abs()
)
residual_results["absolute_percentage_error"] = (
    residual_results["absolute_error"]
    / residual_results[TARGET_COLUMN]
    * 100
)

if TARGET_COLUMN in residual_results.columns:
    residual_results["price_bucket"] = pd.qcut(
        residual_results[TARGET_COLUMN],
        q=int(RESIDUAL_CONFIG.get("price_quantiles", 4)),
        duplicates="drop",
    )

if "surface_area" in residual_results.columns:
    residual_results["surface_bucket"] = pd.qcut(
        residual_results["surface_area"],
        q=int(
            RESIDUAL_CONFIG.get("surface_quantiles", 4)
        ),
        duplicates="drop",
    )


def residual_segment_summary(
    residual_df: pd.DataFrame,
    segment_column: str,
    *,
    minimum_count: int,
) -> pd.DataFrame:
    if segment_column not in residual_df.columns:
        return pd.DataFrame()

    return (
        residual_df.groupby(segment_column, dropna=False)
        .agg(
            count=("residual", "size"),
            mean_residual=("residual", "mean"),
            median_residual=("residual", "median"),
            mae=("absolute_error", "mean"),
            mape_pct=(
                "absolute_percentage_error",
                "mean",
            ),
        )
        .query(f"count >= {minimum_count}")
        .sort_values("mae", ascending=False)
    )


minimum_segment_count = int(
    RESIDUAL_CONFIG.get("minimum_segment_count", 5)
)

for segment_column in RESIDUAL_CONFIG.get(
    "segments",
    [
        "region",
        "property_type",
        "number_of_bedrooms",
        "price_bucket",
        "surface_bucket",
    ],
):
    print(f"Residuals by {segment_column}")
    display(
        residual_segment_summary(
            residual_results,
            segment_column,
            minimum_count=minimum_segment_count,
        )
    )

worst_n = int(
    RESIDUAL_CONFIG.get("worst_predictions_n", 30)
)
display(
    residual_results.sort_values(
        "absolute_error",
        ascending=False,
    ).head(worst_n)
)

plt.figure(figsize=(9, 6))
plt.scatter(
    residual_results["predicted_price"],
    residual_results["residual"],
    alpha=0.5,
)
plt.axhline(0, linestyle="--")
plt.xlabel("Predicted price")
plt.ylabel("Residual: actual - predicted")
plt.title("Residuals versus predicted price")
plt.tight_layout()
plt.show()

plt.figure(figsize=(9, 6))
plt.hist(
    residual_results["residual"],
    bins=int(RESIDUAL_CONFIG.get("histogram_bins", 20)),
)
plt.axvline(0, linestyle="--")
plt.xlabel("Residual: actual - predicted")
plt.ylabel("Count")
plt.title("Residual distribution")
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 23. SHAP / feature importance for the selected model only

# COMMAND ----------

shap_table = pd.DataFrame()

if run_shap and SHAP_CONFIG.get("enabled", False):
    try:
        import shap

        sample_size = min(
            int(SHAP_CONFIG.get("sample_size", 200)),
            len(test_df),
        )
        sample = test_df.sample(
            n=sample_size,
            random_state=RANDOM_STATE,
        )

        if selected_model_family == "catboost":
            X_shap = prepare_catboost_frame(
                sample,
                selected_features,
                selected_categorical_features,
            )
            explainer = shap.TreeExplainer(final_model)
            shap_values = explainer(X_shap)

            shap.plots.bar(
                shap_values,
                max_display=int(
                    SHAP_CONFIG.get("max_display", 20)
                ),
            )
            shap.plots.beeswarm(
                shap_values,
                max_display=int(
                    SHAP_CONFIG.get("max_display", 20)
                ),
            )

            shap_table = (
                pd.DataFrame(
                    {
                        "feature": X_shap.columns,
                        "mean_absolute_shap": np.abs(
                            shap_values.values
                        ).mean(axis=0),
                    }
                )
                .sort_values(
                    "mean_absolute_shap",
                    ascending=False,
                )
                .reset_index(drop=True)
            )

        elif selected_model_family == "random_forest":
            fitted_preprocessor = final_model.named_steps[
                "preprocessor"
            ]
            fitted_estimator = final_model.named_steps["model"]

            transformed = fitted_preprocessor.transform(
                sample[selected_features]
            )
            if hasattr(transformed, "toarray"):
                transformed = transformed.toarray()

            feature_names = (
                fitted_preprocessor.get_feature_names_out()
            )
            feature_names = [
                name.replace("numeric__", "")
                .replace("categorical__", "")
                for name in feature_names
            ]

            transformed_df = pd.DataFrame(
                transformed,
                columns=feature_names,
                index=sample.index,
            )

            explainer = shap.TreeExplainer(
                fitted_estimator
            )
            shap_values = explainer(
                transformed_df,
                check_additivity=False,
            )

            shap.plots.bar(
                shap_values,
                max_display=int(
                    SHAP_CONFIG.get("max_display", 20)
                ),
            )
            shap.plots.beeswarm(
                shap_values,
                max_display=int(
                    SHAP_CONFIG.get("max_display", 20)
                ),
            )

            shap_table = (
                pd.DataFrame(
                    {
                        "feature": transformed_df.columns,
                        "mean_absolute_shap": np.abs(
                            shap_values.values
                        ).mean(axis=0),
                    }
                )
                .sort_values(
                    "mean_absolute_shap",
                    ascending=False,
                )
                .reset_index(drop=True)
            )

        display(shap_table)

    except Exception as exc:
        print(f"SHAP skipped: {exc}")
else:
    print("SHAP disabled.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 24. Quality gates for the selected candidate

# COMMAND ----------

segment_region = (
    evaluate_by_segment(
        residual_results,
        TARGET_COLUMN,
        "predicted_price",
        "region",
    )
    if "region" in residual_results.columns
    else pd.DataFrame()
)

segment_property_type = (
    evaluate_by_segment(
        residual_results,
        TARGET_COLUMN,
        "predicted_price",
        "property_type",
    )
    if "property_type" in residual_results.columns
    else pd.DataFrame()
)

residual_results["price_category"] = residual_results[
    TARGET_COLUMN
].apply(price_category)

segment_price = evaluate_by_segment(
    residual_results,
    TARGET_COLUMN,
    "predicted_price",
    "price_category",
)

within_10pct = float(
    (
        residual_results["absolute_percentage_error"]
        <= 10
    ).mean()
)

gate_summary = {
    "test_metrics": {
        "mae": test_model_metrics["mae"],
        "pct_within_10pct": within_10pct,
    },
    "baseline_metrics": {
        "mae": test_baseline_metrics["mae"],
    },
    "beats_baseline": (
        test_model_metrics["mae"]
        < test_baseline_metrics["mae"]
    ),
    "walk_forward_model_mae_mean": (
        test_model_metrics["mae"]
    ),
    "walk_forward_baseline_mae_mean": (
        test_baseline_metrics["mae"]
    ),
}

candidate_gate_result = evaluate_training_gates(
    gate_summary,
    segment_region,
    segment_property_type,
    segment_price,
    gates=gates_config,
)

display(format_gate_report(candidate_gate_result))

promotion_eligible = False

print(
    "Promotion eligible from this notebook: "
    f"{promotion_eligible}"
)
print(
    "Reason: lab lane never registers models; "
    "official pipeline remains authoritative."
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 25. MLflow lab report

# COMMAND ----------

def _log_dataframe(
    frame: pd.DataFrame,
    artifact_file: str,
) -> None:
    if frame is None or frame.empty:
        return

    mlflow.log_table(
        frame.reset_index(drop=True),
        artifact_file,
    )


lab_beats_baseline = (
    test_model_metrics["mae"] < test_baseline_metrics["mae"]
)

with mlflow.start_run(
    run_name="enterprise_eda_feature_discovery"
) as lab_run:
    mlflow.set_tags(
        {
            "training_lane": "experiment",
            "lane": "lab",
            "register_model": "false",
            "beats_baseline": str(lab_beats_baseline),
            "gates_passed": str(candidate_gate_result.passed),
            "gate_failures": (
                "; ".join(candidate_gate_result.failures)
                if candidate_gate_result.failures
                else "none"
            ),
            "git_commit": lab_git_commit,
            "data_source": data_source,
            "catalog": catalog,
            "selected_model_family": selected_model_family,
            "selected_feature_set": selected_candidate[
                "feature_set"
            ],
            "final_test_used_once": "true",
        }
    )

    mlflow.log_dict(
        LAB_CONFIG,
        "config/eda_lab_enterprise.json",
    )
    mlflow.log_dict(
        dq_summary,
        "data_quality/data_quality_summary.json",
    )
    mlflow.log_dict(
        {
            "train_rows": len(train_df),
            "validation_rows": len(validation_df),
            "test_rows": len(test_df),
            "train_end_date": str(split.train_end_date),
            "validation_end_date": str(
                split.validation_end_date
            ),
        },
        "splits/chronological_split.json",
    )
    mlflow.log_dict(
        {
            "validation_baseline": (
                validation_baseline_metrics
            ),
            "test_baseline": test_baseline_metrics,
            "test_model": test_model_metrics,
            "test_mae_improvement": (
                test_mae_improvement
            ),
            "test_mae_improvement_pct": (
                test_mae_improvement_pct
            ),
        },
        "evaluation/final_metrics.json",
    )
    mlflow.log_dict(
        {
            "passed": candidate_gate_result.passed,
            "failures": candidate_gate_result.failures,
            "details": candidate_gate_result.details,
        },
        "evaluation/gate_report.json",
    )

    _log_dataframe(
        cleaning_funnel,
        "data_quality/cleaning_funnel.json",
    )
    _log_dataframe(
        duplicate_summary,
        "data_quality/duplicate_summary.json",
    )
    _log_dataframe(
        feature_inventory,
        "eda/feature_inventory.json",
    )
    _log_dataframe(
        numeric_bivariate
        if "numeric_bivariate" in globals()
        else pd.DataFrame(),
        "eda/numeric_bivariate.json",
    )
    _log_dataframe(
        categorical_bivariate
        if "categorical_bivariate" in globals()
        else pd.DataFrame(),
        "eda/categorical_bivariate.json",
    )
    _log_dataframe(
        hypothesis_results,
        "eda/business_hypotheses.json",
    )
    _log_dataframe(
        single_feature_results,
        "experiments/single_feature_screen.json",
    )
    _log_dataframe(
        rf_feature_matrix,
        "experiments/feature_matrix.json",
    )
    _log_dataframe(
        best_by_depth,
        "experiments/best_by_depth.json",
    )
    _log_dataframe(
        top_group_frequency,
        "experiments/top_group_frequency.json",
    )
    _log_dataframe(
        model_family_results,
        "experiments/model_family_comparison.json",
    )
    _log_dataframe(
        final_test_comparison,
        "evaluation/final_test_comparison.json",
    )
    _log_dataframe(
        residual_results,
        "evaluation/test_residuals.json",
    )
    _log_dataframe(
        shap_table,
        "explainability/shap_importance.json",
    )

    mlflow.log_metrics(
        {
            "test_mae": test_model_metrics["mae"],
            "test_rmse": test_model_metrics["rmse"],
            "test_r2": test_model_metrics["r2"],
            "test_mape_pct": test_model_metrics["mape_pct"],
            "baseline_mae": test_baseline_metrics["mae"],
            "baseline_rmse": test_baseline_metrics["rmse"],
            "validation_baseline_mae": (
                validation_baseline_metrics["mae"]
            ),
            "test_baseline_mae": (
                test_baseline_metrics["mae"]
            ),
            "test_model_mae": test_model_metrics["mae"],
            "test_model_rmse": test_model_metrics["rmse"],
            "test_model_r2": test_model_metrics["r2"],
            "test_model_mape_pct": (
                test_model_metrics["mape_pct"]
            ),
            "beats_baseline": 1.0 if lab_beats_baseline else 0.0,
            "gates_passed": (
                1.0 if candidate_gate_result.passed else 0.0
            ),
            "mae_improvement_pct": test_mae_improvement_pct,
            "test_mae_improvement_pct": (
                test_mae_improvement_pct
            ),
        }
    )

    lab_run_id = lab_run.info.run_id

print(f"Enterprise lab MLflow run ID: {lab_run_id}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 26. Optional official-pipeline dry run
# MAGIC
# MAGIC This remains lab-only and does not register a UC model.

# COMMAND ----------

if run_full_train_dry_run:
    deep_config = training_config_from_job_params(
        enable_tuning=(
            "true" if enable_tuning else "false"
        ),
        enable_ablation=(
            "true" if enable_ablation else "false"
        ),
        enable_explainability=(
            "true" if enable_explainability else "false"
        ),
    )

    output_dir = Path("/tmp/enterprise_lab_dry_run")

    train(
        model_df,
        output_dir=output_dir,
        training_config=deep_config,
        catalog=catalog,
        register_model=False,
        enforce_gates=False,
        git_commit=(
            lab_git_commit
            if lab_git_commit not in ("unknown", "none", "")
            else None
        ),
        data_source=f"enterprise-lab:{data_source}",
        mlflow_experiment_name=mlflow_experiment,
    )

    print(
        "Official-pipeline dry run complete. "
        "No Unity Catalog registration performed."
    )
else:
    print("Official-pipeline dry run disabled.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 27. Recent MLflow runs

# COMMAND ----------

display(search_recent_runs(mlflow_experiment, n=10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 28. Final checklist
# MAGIC
# MAGIC Before promotion through official CI:
# MAGIC
# MAGIC - Commit `ml/config/eda_lab_enterprise.yaml`
# MAGIC - Commit feature-engineering changes under `ml/src/`
# MAGIC - Confirm exact-duplicate handling in Silver/Gold
# MAGIC - Confirm point-in-time correctness of historical features
# MAGIC - Confirm selected features are available at inference time
# MAGIC - Confirm final test was not used for selection
# MAGIC - Run official experiment pipeline with gates enforced
# MAGIC - Promote only an official run with `gates_passed=1`

# COMMAND ----------

print("Enterprise lab playbook complete.")
print(
    "Selected candidate: "
    f"{selected_model_family} — "
    f"{selected_candidate['feature_set']}"
)
print(
    "Final test MAE: "
    f"{test_model_metrics['mae']:,.2f}"
)
print(
    "Final test improvement versus baseline: "
    f"{test_mae_improvement_pct:.2f}%"
)
