# Databricks notebook source
# MAGIC %md
# MAGIC # House Price Experiment Notebook
# MAGIC
# MAGIC **Assessment workflow:** data quality → EDA → hypotheses → feature engineering →
# MAGIC chronological train/validation/test → business baseline → single-feature screen →
# MAGIC feature-group matrix → model comparison → final test → residuals → SHAP → MLflow.
# MAGIC
# MAGIC This notebook is intentionally self-contained for assessment review. It imports
# MAGIC production logic where consistency matters (data loading, baseline, metrics,
# MAGIC point-in-time historical features) and keeps experiment flow inline.
# MAGIC
# MAGIC **Run order:** sections 0–16 for validation work, then set `UNLOCK_FINAL_TEST = True`
# MAGIC in section 1 for sections 17–22 (final test, residuals, SHAP, MLflow).
# MAGIC
# MAGIC Edit the **Run settings** constants in section 1 — no widgets.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 0. Install project wheel
# MAGIC
# MAGIC Edit `WHEEL_PATH` below if `house_price_ml` is not already on the cluster.

# COMMAND ----------

import importlib.util
import subprocess
import sys

# Edit this path when the package is not pre-installed on the cluster.
WHEEL_PATH = ""

PACKAGE_ROOT = "house_price_ml"
if importlib.util.find_spec(PACKAGE_ROOT) is None:
    wheel_path = WHEEL_PATH.strip()
    if not wheel_path:
        raise ModuleNotFoundError(
            "house_price_ml is unavailable. Set WHEEL_PATH above to a deployed .whl."
        )
    print(f"Installing wheel from: {wheel_path}")
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", wheel_path, "--force-reinstall", "-q"]
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
# MAGIC ## 1. Imports and run settings
# MAGIC
# MAGIC Edit the constants below, then run this cell before the rest of the notebook.

# COMMAND ----------

import itertools
import json
import time
import warnings
from dataclasses import dataclass
from typing import Any, Sequence

import mlflow
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

try:
    from catboost import CatBoostRegressor
except Exception:
    CatBoostRegressor = None

try:
    import shap
except Exception:
    shap = None

from house_price_ml.config.training_config import load_training_config
from house_price_ml.evaluation.metrics import compute_metrics as production_compute_metrics
from house_price_ml.features.historical import add_historical_market_features
from house_price_ml.features.pipeline import raw_to_feature_frame
from house_price_ml.jobs.experiment_lab import (
    configure_lab_mlflow,
    data_quality_summary,
    load_training_frame_from_catalog,
    load_training_frame_sample,
)
from house_price_ml.models.baseline import BusinessBaseline
from house_price_ml.models.train import _git_commit
from house_price_ml.serving.mlflow_model import build_sklearn_pipeline

warnings.filterwarnings("ignore")

# --- Run settings (edit here) ---
CATALOG = "house_price_staging"
DATA_SOURCE = "delta"  # "delta" or "sample"
SAMPLE_PROFILE = "demo"  # used when DATA_SOURCE == "sample"
SAMPLE_ROWS = None  # e.g. 5000; None uses the profile default
RUN_CATBOOST = True
RUN_SHAP = False
UNLOCK_FINAL_TEST = False  # set True for sections 17–22
MLFLOW_EXPERIMENT = "/Shared/house_price_prediction_assessment"
GIT_COMMIT = _git_commit(None)

TARGET_COLUMN = "label_sale_price"
DATE_COLUMN = "sale_date"
ID_COLUMN = "listing_id"

TRAIN_FRACTION = 0.70
VALIDATION_FRACTION = 0.15
RANDOM_STATE = 42
MAX_FEATURE_GROUP_DEPTH = 4
MAX_FEATURE_EXPERIMENTS = 500
RF_SEARCH_ESTIMATORS = 200
RF_COMPARISON_ESTIMATORS = 500
RF_FINAL_ESTIMATORS = 500
RF_MIN_SAMPLES_LEAF = 3
CATBOOST_ITERATIONS = 1000
CATBOOST_EARLY_STOPPING_ROUNDS = 75
MIN_CATEGORY_GROUP_SIZE = 5
MIN_RESIDUAL_SEGMENT_SIZE = 5
CANDIDATE_MAE_TOLERANCE = 0.02
FINAL_MODEL_MAE_TOLERANCE = 0.01
TOP_FEATURE_SETS_FOR_MODEL_COMPARISON = 3

EXCLUDED_FEATURES = {
    TARGET_COLUMN,
    DATE_COLUMN,
    ID_COLUMN,
    "asking_price",
    "sale_price",
    "sale_price_per_sqm",
}

DUPLICATE_KEY = [
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
]

FEATURE_GROUPS = {
    "surface": ["surface_area"],
    "region": ["region"],
    "coordinates": ["latitude", "longitude"],
    "property_type": ["property_type"],
    "rooms": ["number_of_rooms", "number_of_bedrooms"],
    "age": ["build_year", "house_age", "is_new_build"],
    "property_characteristics": ["energy_label", "garden"],
    "engineered_space": [
        "surface_per_room",
        "surface_per_bedroom",
        "bedroom_ratio",
        "room_density",
        "non_bedroom_rooms",
        "surface_x_garden",
    ],
    "postcode": ["postcode_prefix"],
    "historical_market": [
        "historic_region_median_price_per_sqm",
        "historic_region_property_median_price_per_sqm",
        "historic_region_12m_median_price_per_sqm",
        "historic_region_property_12m_median_price_per_sqm",
        "historic_region_sale_count",
        "historic_region_property_sale_count",
        "historic_region_12m_sale_count",
        "historic_region_property_12m_sale_count",
        "historic_region_value_estimate",
        "historic_region_property_value_estimate",
        "historic_region_12m_value_estimate",
        "historic_region_property_12m_value_estimate",
    ],
}

configure_lab_mlflow(MLFLOW_EXPERIMENT)

print(f"Catalog: {CATALOG}")
print(f"Data source: {DATA_SOURCE}")
print(f"MLflow experiment: {MLFLOW_EXPERIMENT}")
print(f"Git commit: {GIT_COMMIT}")
print("Model registration: disabled")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Utility functions

# COMMAND ----------

@dataclass(frozen=True)
class ChronologicalSplit:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    train_end_date: pd.Timestamp
    validation_end_date: pd.Timestamp


def calculate_metrics(actual: Sequence[float], predicted: Sequence[float]) -> dict[str, float]:
    """Assessment metrics aligned with production compute_metrics, plus R²."""
    actual_array = np.asarray(actual, dtype=float)
    predicted_array = np.asarray(predicted, dtype=float)
    metrics = production_compute_metrics(actual_array, predicted_array)
    return {
        "mae": float(metrics["mae"]),
        "rmse": float(metrics["rmse"]),
        "bias": float(metrics["bias"]),
        "mape_pct": float(metrics["mape"]),
        "mdape_pct": float(metrics["mdape"]),
        "pct_within_10pct": float(metrics["pct_within_10pct"]),
        "median_ae": float(np.median(np.abs(actual_array - predicted_array))),
        "r2": float(r2_score(actual_array, predicted_array)),
    }


def infer_feature_inventory(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for column in df.columns:
        series = df[column]
        unique_count = int(series.nunique(dropna=False))
        if column == TARGET_COLUMN:
            role = "target"
        elif column == DATE_COLUMN:
            role = "date"
        elif column == ID_COLUMN:
            role = "identifier"
        elif column in EXCLUDED_FEATURES:
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
                "missing_count": int(series.isna().sum()),
                "missing_pct": float(series.isna().mean() * 100),
                "unique_count": unique_count,
                "constant_column": unique_count <= 1,
                "high_cardinality": unique_count > max(20, int(len(df) * 0.50)),
            }
        )
    return pd.DataFrame(rows)


def exact_duplicate_report(
    df: pd.DataFrame,
    duplicate_key: Sequence[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    available_key = [column for column in duplicate_key if column in df.columns]
    if not available_key:
        raise ValueError("No duplicate-key columns are available.")
    duplicate_mask = df.duplicated(subset=available_key, keep=False)
    groups = (
        df.loc[duplicate_mask]
        .groupby(available_key, dropna=False)
        .size()
        .reset_index(name="duplicate_count")
        .sort_values("duplicate_count", ascending=False)
    )
    summary = pd.DataFrame(
        [
            {"metric": "raw_rows", "value": len(df)},
            {"metric": "rows_in_duplicate_groups", "value": int(duplicate_mask.sum())},
            {"metric": "exact_duplicate_groups", "value": len(groups)},
            {
                "metric": "unique_rows_after_deduplication",
                "value": len(df.drop_duplicates(subset=available_key, keep="first")),
            },
            {"metric": "duplicate_group_row_rate", "value": float(duplicate_mask.mean())},
        ]
    )
    return summary, groups


def clean_dataframe(
    df: pd.DataFrame,
    duplicate_key: Sequence[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    result = df.copy()
    funnel = [{"step": "raw", "removed": 0, "remaining": len(result)}]
    result[DATE_COLUMN] = pd.to_datetime(result[DATE_COLUMN], errors="coerce")
    for column in [
        "surface_area",
        "number_of_rooms",
        "number_of_bedrooms",
        "build_year",
        "latitude",
        "longitude",
        TARGET_COLUMN,
    ]:
        if column in result.columns:
            result[column] = pd.to_numeric(result[column], errors="coerce")
    result = result.replace([np.inf, -np.inf], np.nan)
    required = [
        column
        for column in [DATE_COLUMN, TARGET_COLUMN, "surface_area", "region"]
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
    valid = pd.Series(True, index=result.index)
    if TARGET_COLUMN in result.columns:
        valid &= result[TARGET_COLUMN] > 0
    if "surface_area" in result.columns:
        valid &= result["surface_area"] > 0
    before = len(result)
    result = result.loc[valid].copy()
    funnel.append(
        {
            "step": "drop_invalid_values",
            "removed": before - len(result),
            "remaining": len(result),
        }
    )
    available_key = [column for column in duplicate_key if column in result.columns]
    before = len(result)
    result = result.drop_duplicates(subset=available_key, keep="first").copy()
    funnel.append(
        {
            "step": "drop_exact_duplicates",
            "removed": before - len(result),
            "remaining": len(result),
        }
    )
    sort_columns = [column for column in [DATE_COLUMN, ID_COLUMN] if column in result.columns]
    result = result.sort_values(sort_columns).reset_index(drop=True)
    funnel.append({"step": "final", "removed": 0, "remaining": len(result)})
    return result, pd.DataFrame(funnel)


def add_general_features(df: pd.DataFrame) -> pd.DataFrame:
    from house_price_ml.features.energy import energy_label_to_score
    from house_price_ml.features.geo import distance_to_city_centre

    result = df.copy()
    snapshot_col = (
        "feature_snapshot_date"
        if "feature_snapshot_date" in result.columns
        else DATE_COLUMN
    )
    result[snapshot_col] = pd.to_datetime(result[snapshot_col], errors="coerce")
    rooms_safe = result["number_of_rooms"].replace(0, np.nan)
    bedrooms_safe = result["number_of_bedrooms"].replace(0, np.nan)
    surface_safe = result["surface_area"].replace(0, np.nan)
    result["house_age"] = result[snapshot_col].dt.year - result["build_year"]
    result.loc[(result["house_age"] < 0) | (result["house_age"] > 500), "house_age"] = np.nan
    result["is_new_build"] = result["house_age"].between(0, 5).astype("string")
    result["surface_per_room"] = result["surface_area"] / rooms_safe
    result["surface_per_bedroom"] = result["surface_area"] / bedrooms_safe
    result["bedroom_ratio"] = result["number_of_bedrooms"] / rooms_safe
    result["room_density"] = result["number_of_rooms"] / surface_safe
    result["non_bedroom_rooms"] = result["number_of_rooms"] - result["number_of_bedrooms"]
    garden_numeric = result["garden"].fillna(False).astype(bool).astype(int)
    result["surface_x_garden"] = result["surface_area"] * garden_numeric
    result["postcode_prefix"] = (
        result["postcode"].astype("string").str.extract(r"(\d{2})", expand=False)
    )
    result["energy_label_score"] = result["energy_label"].map(energy_label_to_score)
    result["dist_to_city_centre_km"] = result.apply(
        lambda row: distance_to_city_centre(
            row["region"],
            float(row["latitude"]),
            float(row["longitude"]),
        ),
        axis=1,
    )
    result["month"] = result[snapshot_col].dt.month
    result["quarter"] = result[snapshot_col].dt.quarter
    return result.replace([np.inf, -np.inf], np.nan)


def chronological_split(
    df: pd.DataFrame,
    train_fraction: float,
    validation_fraction: float,
) -> ChronologicalSplit:
    working = (
        df.copy()
        .sort_values([column for column in [DATE_COLUMN, ID_COLUMN] if column in df.columns])
        .reset_index(drop=True)
    )
    unique_dates = pd.Series(working[DATE_COLUMN].dropna().sort_values().unique())
    if len(unique_dates) < 3:
        raise ValueError("At least three unique dates are required.")
    train_index = min(
        len(unique_dates) - 3,
        max(0, int(len(unique_dates) * train_fraction) - 1),
    )
    validation_index = min(
        len(unique_dates) - 2,
        max(
            train_index + 1,
            int(len(unique_dates) * (train_fraction + validation_fraction)) - 1,
        ),
    )
    train_end = pd.Timestamp(unique_dates.iloc[train_index])
    validation_end = pd.Timestamp(unique_dates.iloc[validation_index])
    train = working[working[DATE_COLUMN] <= train_end].copy()
    validation = working[
        (working[DATE_COLUMN] > train_end)
        & (working[DATE_COLUMN] <= validation_end)
    ].copy()
    test = working[working[DATE_COLUMN] > validation_end].copy()
    if min(len(train), len(validation), len(test)) == 0:
        raise ValueError("Chronological split produced an empty partition.")
    return ChronologicalSplit(train, validation, test, train_end, validation_end)


def build_random_forest_pipeline(
    numeric_features: Sequence[str],
    categorical_features: Sequence[str],
    n_estimators: int,
) -> Pipeline:
    transformers = []
    if numeric_features:
        transformers.append(
            (
                "numeric",
                Pipeline([("imputer", SimpleImputer(strategy="median"))]),
                list(numeric_features),
            )
        )
    if categorical_features:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "onehot",
                            OneHotEncoder(handle_unknown="ignore", min_frequency=3),
                        ),
                    ]
                ),
                list(categorical_features),
            )
        )
    return Pipeline(
        [
            ("preprocessor", ColumnTransformer(transformers=transformers, remainder="drop")),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=n_estimators,
                    min_samples_leaf=RF_MIN_SAMPLES_LEAF,
                    max_features="sqrt",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def split_feature_types(
    features: Sequence[str],
    inventory: pd.DataFrame,
) -> tuple[list[str], list[str]]:
    role_by_column = dict(zip(inventory["column"], inventory["role"]))
    numeric = [feature for feature in features if role_by_column.get(feature) == "numeric"]
    categorical = [
        feature for feature in features if role_by_column.get(feature) == "categorical"
    ]
    return numeric, categorical


def prepare_catboost_data(
    df: pd.DataFrame,
    features: Sequence[str],
    categorical_features: Sequence[str],
) -> pd.DataFrame:
    result = df[list(features)].copy()
    categorical_set = set(categorical_features)
    for feature in features:
        if feature in categorical_set:
            result[feature] = (
                result[feature].astype("string").fillna("__MISSING__").astype(str)
            )
        else:
            result[feature] = pd.to_numeric(result[feature], errors="coerce")
    return result


def validate_feature_groups(
    configured_groups: dict[str, list[str]],
    df: pd.DataFrame,
) -> tuple[dict[str, list[str]], pd.DataFrame]:
    valid_groups = {}
    rows = []
    for group, features in configured_groups.items():
        available = [feature for feature in features if feature in df.columns]
        missing = [feature for feature in features if feature not in df.columns]
        rows.append(
            {
                "group": group,
                "configured_count": len(features),
                "available_count": len(available),
                "available_features": available,
                "missing_features": missing,
                "enabled": bool(available),
            }
        )
        if available:
            valid_groups[group] = available
    return valid_groups, pd.DataFrame(rows)


def generate_group_combinations(
    feature_groups: dict[str, list[str]],
    max_depth: int,
) -> list[tuple[str, ...]]:
    combinations = []
    group_names = list(feature_groups)
    for depth in range(1, min(max_depth, len(group_names)) + 1):
        combinations.extend(itertools.combinations(group_names, depth))
    if len(combinations) > MAX_FEATURE_EXPERIMENTS:
        raise ValueError(
            f"Generated {len(combinations)} experiments; maximum is {MAX_FEATURE_EXPERIMENTS}."
        )
    return combinations


def features_from_groups(
    groups: Sequence[str],
    feature_groups: dict[str, list[str]],
) -> list[str]:
    features = []
    for group in groups:
        features.extend(feature_groups[group])
    return list(dict.fromkeys(features))


def residual_segment_summary(
    residual_df: pd.DataFrame,
    segment: str,
    minimum_count: int,
) -> pd.DataFrame:
    if segment not in residual_df.columns:
        return pd.DataFrame()
    return (
        residual_df.groupby(segment, dropna=False)
        .agg(
            count=("residual", "size"),
            mean_residual=("residual", "mean"),
            median_residual=("residual", "median"),
            mae=("absolute_error", "mean"),
            mape_pct=("absolute_percentage_error", "mean"),
        )
        .query(f"count >= {minimum_count}")
        .sort_values("mae", ascending=False)
    )


def mlflow_safe_frame(df: pd.DataFrame) -> pd.DataFrame:
    safe = df.reset_index(drop=True).copy()
    for column in safe.columns:
        safe[column] = safe[column].map(
            lambda value: (
                json.dumps(value, default=str)
                if isinstance(value, (list, tuple, dict, set))
                else str(value)
                if isinstance(value, (pd.Interval, pd.Timestamp))
                else value
            )
        )
    return safe

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Load data

# COMMAND ----------

rejected_df = None

if DATA_SOURCE == "delta":
    source_df = spark.table(f"{CATALOG}.silver.listings_clean").toPandas()
    model_df = load_training_frame_from_catalog(spark, CATALOG)
    try:
        rejected_df = spark.table(f"{CATALOG}.silver.listings_rejected").toPandas()
    except Exception:
        rejected_df = None
else:
    model_df = load_training_frame_sample(profile=SAMPLE_PROFILE, rows=SAMPLE_ROWS)
    source_df = model_df.copy()

print(f"Source rows: {len(source_df):,}")
print(f"Model rows: {len(model_df):,}")
display(source_df.head())
display(model_df.head())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Data quality and cleaning

# COMMAND ----------

duplicate_summary, duplicate_groups = exact_duplicate_report(source_df, DUPLICATE_KEY)
display(duplicate_summary)
display(duplicate_groups.head(30))

dq_summary = data_quality_summary(source_df, rejected_df)
display(
    pd.DataFrame(
        [
            {"metric": key, "value": value}
            for key, value in dq_summary.items()
            if key != "null_rates"
        ]
    )
)
if dq_summary.get("null_rates"):
    display(
        pd.DataFrame(
            dq_summary["null_rates"].items(),
            columns=["column", "null_rate"],
        )
    )
if rejected_df is not None and len(rejected_df):
    display(rejected_df.head(10))

clean_source_df, source_cleaning_funnel = clean_dataframe(source_df, DUPLICATE_KEY)
display(source_cleaning_funnel)

model_duplicate_key = [column for column in DUPLICATE_KEY if column in model_df.columns]
model_df, model_cleaning_funnel = clean_dataframe(model_df, model_duplicate_key)
display(model_cleaning_funnel)

print(f"Effective model sample size: {len(model_df):,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Feature engineering

# COMMAND ----------

model_df = add_general_features(model_df)
model_df = add_historical_market_features(
    model_df,
    rolling_window_days=365,
    min_history_count=5,
)

feature_inventory = infer_feature_inventory(model_df)
display(feature_inventory.sort_values(["role", "missing_pct", "unique_count"]))

numeric_features = feature_inventory.loc[
    feature_inventory["role"] == "numeric", "column"
].tolist()
categorical_features = feature_inventory.loc[
    feature_inventory["role"] == "categorical", "column"
].tolist()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Univariate EDA

# COMMAND ----------

numeric_profile = model_df[numeric_features].describe(
    percentiles=[0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]
).T
numeric_profile["missing_count"] = model_df[numeric_features].isna().sum()
numeric_profile["missing_pct"] = model_df[numeric_features].isna().mean() * 100
numeric_profile["skew"] = model_df[numeric_features].skew(numeric_only=True)
display(numeric_profile)

for column in numeric_features:
    values = model_df[column].dropna()
    if values.empty:
        continue
    plt.figure(figsize=(9, 4))
    plt.hist(values, bins=30)
    plt.axvline(values.median(), linestyle="--", label="Median")
    plt.xlabel(column)
    plt.ylabel("Count")
    plt.title(f"Distribution of {column}")
    plt.legend()
    plt.tight_layout()
    plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Categorical EDA

# COMMAND ----------

categorical_summary_rows = []
for column in categorical_features:
    counts = model_df[column].astype("string").fillna("__MISSING__").value_counts()
    categorical_summary_rows.append(
        {
            "feature": column,
            "unique_count": len(counts),
            "missing_count": int(model_df[column].isna().sum()),
            "missing_pct": float(model_df[column].isna().mean() * 100),
            "most_common": str(counts.index[0]) if len(counts) else None,
            "most_common_count": int(counts.iloc[0]) if len(counts) else 0,
        }
    )

categorical_summary = pd.DataFrame(categorical_summary_rows)
display(categorical_summary)

for column in categorical_features:
    counts = (
        model_df[column]
        .astype("string")
        .fillna("__MISSING__")
        .value_counts()
        .head(15)
        .sort_values()
    )
    if counts.empty:
        continue
    plt.figure(figsize=(9, max(4, len(counts) * 0.35)))
    plt.barh(counts.index.astype(str), counts.values)
    plt.xlabel("Count")
    plt.ylabel(column)
    plt.title(f"Most common categories: {column}")
    plt.tight_layout()
    plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Bivariate EDA and correlation

# COMMAND ----------

numeric_target_rows = []
for feature in numeric_features:
    pair = model_df[[feature, TARGET_COLUMN]].dropna()
    if len(pair) < 3:
        continue
    correlation = pair[feature].corr(pair[TARGET_COLUMN])
    numeric_target_rows.append(
        {
            "feature": feature,
            "count": len(pair),
            "pearson_correlation": correlation,
            "correlation_squared": correlation**2 if pd.notna(correlation) else np.nan,
        }
    )

numeric_target_report = pd.DataFrame(numeric_target_rows).sort_values(
    "pearson_correlation",
    key=lambda values: values.abs(),
    ascending=False,
)
display(numeric_target_report)

categorical_target_rows = []
for feature in categorical_features:
    grouped = (
        model_df.assign(
            **{
                feature: model_df[feature]
                .astype("string")
                .fillna("__MISSING__")
            }
        )
        .groupby(feature)[TARGET_COLUMN]
        .agg(count="size", mean="mean", median="median", std="std")
        .query(f"count >= {MIN_CATEGORY_GROUP_SIZE}")
        .reset_index()
    )
    for _, row in grouped.iterrows():
        categorical_target_rows.append(
            {
                "feature": feature,
                "category": row[feature],
                "count": int(row["count"]),
                "mean_target": float(row["mean"]),
                "median_target": float(row["median"]),
                "std_target": float(row["std"]) if pd.notna(row["std"]) else np.nan,
            }
        )

categorical_target_report = pd.DataFrame(categorical_target_rows)
display(categorical_target_report)

correlation_columns = [
    column
    for column in numeric_features + [TARGET_COLUMN]
    if column in model_df.columns
]
correlation_matrix = model_df[correlation_columns].corr(numeric_only=True)
display(correlation_matrix)

plt.figure(
    figsize=(
        max(8, len(correlation_columns) * 0.7),
        max(7, len(correlation_columns) * 0.7),
    )
)
image = plt.imshow(correlation_matrix, aspect="auto", vmin=-1, vmax=1)
plt.colorbar(image, label="Pearson correlation")
plt.xticks(range(len(correlation_columns)), correlation_columns, rotation=90)
plt.yticks(range(len(correlation_columns)), correlation_columns)
plt.title("Numeric correlation matrix")
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Time drift and hypotheses

# COMMAND ----------

monthly_drift = (
    model_df[[DATE_COLUMN, TARGET_COLUMN]]
    .dropna()
    .set_index(DATE_COLUMN)
    .resample("ME")[TARGET_COLUMN]
    .agg(["count", "mean", "median"])
    .reset_index()
)
display(monthly_drift)

plt.figure(figsize=(11, 5))
plt.plot(monthly_drift[DATE_COLUMN], monthly_drift["median"])
plt.xlabel("Month")
plt.ylabel(f"Median {TARGET_COLUMN}")
plt.title("Target drift over time")
plt.tight_layout()
plt.show()

business_hypotheses = pd.DataFrame(
    [
        {
            "hypothesis": "Surface area is a strong price driver.",
            "test": "Correlation and single-feature model",
        },
        {
            "hypothesis": "Property type contributes beyond surface area.",
            "test": "Feature-group matrix",
        },
        {
            "hypothesis": "Historical local market features improve generalization.",
            "test": "Feature-group matrix and residuals",
        },
        {
            "hypothesis": "Energy label and garden matter mainly through interactions.",
            "test": "Feature-group matrix and SHAP",
        },
    ]
)
display(business_hypotheses)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Chronological split

# COMMAND ----------

split = chronological_split(model_df, TRAIN_FRACTION, VALIDATION_FRACTION)
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
# MAGIC ## 11. Business baseline

# COMMAND ----------

validation_baseline = BusinessBaseline().fit(train_df)
validation_baseline_predictions = validation_baseline.predict(validation_df)
validation_baseline_metrics = calculate_metrics(
    validation_df[TARGET_COLUMN],
    validation_baseline_predictions,
)
display(pd.DataFrame([{"model": "BusinessBaseline", **validation_baseline_metrics}]))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 12. Single-feature predictive screen

# COMMAND ----------

candidate_inventory = feature_inventory[
    feature_inventory["role"].isin(["numeric", "categorical"])
].copy()

single_feature_rows = []
for _, item in candidate_inventory.iterrows():
    feature = item["column"]
    feature_type = item["role"]
    numeric = [feature] if feature_type == "numeric" else []
    categorical = [feature] if feature_type == "categorical" else []
    pipeline = build_random_forest_pipeline(
        numeric,
        categorical,
        n_estimators=RF_SEARCH_ESTIMATORS,
    )
    try:
        start = time.perf_counter()
        pipeline.fit(train_df[[feature]], train_df[TARGET_COLUMN])
        predictions = pipeline.predict(validation_df[[feature]])
        elapsed = time.perf_counter() - start
        metrics = calculate_metrics(validation_df[TARGET_COLUMN], predictions)
        single_feature_rows.append(
            {
                "feature": feature,
                "feature_type": feature_type,
                **metrics,
                "mae_improvement_vs_baseline": validation_baseline_metrics["mae"]
                - metrics["mae"],
                "mae_improvement_vs_baseline_pct": (
                    (validation_baseline_metrics["mae"] - metrics["mae"])
                    / validation_baseline_metrics["mae"]
                    * 100
                ),
                "training_seconds": elapsed,
                "status": "ok",
            }
        )
    except Exception as exc:
        single_feature_rows.append(
            {
                "feature": feature,
                "feature_type": feature_type,
                "status": f"error: {exc}",
            }
        )

single_feature_results = pd.DataFrame(single_feature_rows)
if "mae" in single_feature_results.columns:
    single_feature_results = (
        single_feature_results.sort_values(["mae", "r2"], ascending=[True, False])
        .reset_index(drop=True)
    )
    single_feature_results.insert(
        0,
        "rank",
        np.arange(1, len(single_feature_results) + 1),
    )
display(single_feature_results)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 13. Feature-group matrix

# COMMAND ----------

valid_feature_groups, feature_group_inventory = validate_feature_groups(
    FEATURE_GROUPS,
    model_df,
)
display(feature_group_inventory)

feature_group_combinations = generate_group_combinations(
    valid_feature_groups,
    MAX_FEATURE_GROUP_DEPTH,
)
print(f"Generated experiments: {len(feature_group_combinations):,}")

feature_matrix_rows = []
for index, groups in enumerate(feature_group_combinations, start=1):
    features = features_from_groups(groups, valid_feature_groups)
    numeric, categorical = split_feature_types(features, feature_inventory)
    pipeline = build_random_forest_pipeline(
        numeric,
        categorical,
        n_estimators=RF_SEARCH_ESTIMATORS,
    )
    start = time.perf_counter()
    pipeline.fit(train_df[features], train_df[TARGET_COLUMN])
    predictions = pipeline.predict(validation_df[features])
    elapsed = time.perf_counter() - start
    metrics = calculate_metrics(validation_df[TARGET_COLUMN], predictions)
    feature_matrix_rows.append(
        {
            "experiment": " + ".join(groups),
            "groups": tuple(groups),
            "depth": len(groups),
            "number_of_raw_features": len(features),
            "feature_list": features,
            **metrics,
            "mae_improvement_vs_baseline": validation_baseline_metrics["mae"]
            - metrics["mae"],
            "mae_improvement_vs_baseline_pct": (
                (validation_baseline_metrics["mae"] - metrics["mae"])
                / validation_baseline_metrics["mae"]
                * 100
            ),
            "beats_baseline": metrics["mae"] < validation_baseline_metrics["mae"],
            "training_seconds": elapsed,
        }
    )
    if index % 50 == 0 or index == len(feature_group_combinations):
        print(f"Completed: {index}/{len(feature_group_combinations)}")

feature_matrix_results = (
    pd.DataFrame(feature_matrix_rows)
    .sort_values(["mae", "depth"], ascending=[True, True])
    .reset_index(drop=True)
)
feature_matrix_results.insert(0, "rank", np.arange(1, len(feature_matrix_results) + 1))
display(feature_matrix_results.head(50))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 14. Feature-search stability and candidate feature sets

# COMMAND ----------

best_by_depth = (
    feature_matrix_results.sort_values("mae")
    .groupby("depth", as_index=False)
    .first()
    .sort_values("depth")
)
display(best_by_depth)

top_n = min(20, len(feature_matrix_results))
top_results = feature_matrix_results.nsmallest(top_n, "mae")
group_frequency_rows = []
for group in valid_feature_groups:
    share = float(top_results["groups"].apply(lambda groups: group in groups).mean())
    group_frequency_rows.append(
        {
            "group": group,
            "count_in_top_n": int(round(share * top_n)),
            "share_in_top_n": share,
        }
    )
group_frequency = pd.DataFrame(group_frequency_rows).sort_values(
    "share_in_top_n", ascending=False
)
display(group_frequency)

best_feature_mae = float(feature_matrix_results["mae"].min())
candidate_feature_sets = (
    feature_matrix_results[
        feature_matrix_results["mae"]
        <= best_feature_mae * (1 + CANDIDATE_MAE_TOLERANCE)
    ]
    .sort_values(
        ["number_of_raw_features", "depth", "mae"],
        ascending=[True, True, True],
    )
    .head(TOP_FEATURE_SETS_FOR_MODEL_COMPARISON)
    .copy()
)
display(candidate_feature_sets)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 15. Model-family comparison

# COMMAND ----------

model_comparison_rows = []
validation_models = {}

for candidate_index, candidate in candidate_feature_sets.iterrows():
    features = list(candidate["feature_list"])
    numeric, categorical = split_feature_types(features, feature_inventory)
    feature_set_name = candidate["experiment"]

    rf_key = f"random_forest::{candidate_index}::{feature_set_name}"
    rf_model = build_random_forest_pipeline(
        numeric,
        categorical,
        n_estimators=RF_COMPARISON_ESTIMATORS,
    )
    start = time.perf_counter()
    rf_model.fit(train_df[features], train_df[TARGET_COLUMN])
    predictions = rf_model.predict(validation_df[features])
    elapsed = time.perf_counter() - start
    metrics = calculate_metrics(validation_df[TARGET_COLUMN], predictions)
    model_comparison_rows.append(
        {
            "model_key": rf_key,
            "model_family": "random_forest",
            "feature_set": feature_set_name,
            "feature_list": features,
            "categorical_features": categorical,
            "best_iteration": np.nan,
            **metrics,
            "training_seconds": elapsed,
        }
    )
    validation_models[rf_key] = rf_model

    if RUN_CATBOOST and CatBoostRegressor is not None:
        cat_key = f"catboost::{candidate_index}::{feature_set_name}"
        X_train_cat = prepare_catboost_data(train_df, features, categorical)
        X_validation_cat = prepare_catboost_data(validation_df, features, categorical)
        cat_model = CatBoostRegressor(
            iterations=CATBOOST_ITERATIONS,
            learning_rate=0.05,
            depth=6,
            loss_function="MAE",
            eval_metric="MAE",
            cat_features=categorical,
            l2_leaf_reg=5.0,
            random_strength=1.0,
            random_seed=RANDOM_STATE,
            thread_count=-1,
            verbose=False,
            allow_writing_files=False,
            use_best_model=True,
        )
        start = time.perf_counter()
        cat_model.fit(
            X_train_cat,
            train_df[TARGET_COLUMN],
            eval_set=(X_validation_cat, validation_df[TARGET_COLUMN]),
            early_stopping_rounds=CATBOOST_EARLY_STOPPING_ROUNDS,
            verbose=False,
        )
        predictions = cat_model.predict(X_validation_cat)
        elapsed = time.perf_counter() - start
        metrics = calculate_metrics(validation_df[TARGET_COLUMN], predictions)
        best_iteration = cat_model.get_best_iteration()
        best_iteration = (
            int(best_iteration + 1)
            if best_iteration is not None and best_iteration >= 0
            else CATBOOST_ITERATIONS
        )
        model_comparison_rows.append(
            {
                "model_key": cat_key,
                "model_family": "catboost",
                "feature_set": feature_set_name,
                "feature_list": features,
                "categorical_features": categorical,
                "best_iteration": best_iteration,
                **metrics,
                "training_seconds": elapsed,
            }
        )
        validation_models[cat_key] = cat_model

model_comparison_results = (
    pd.DataFrame(model_comparison_rows)
    .sort_values(["mae", "model_family"], ascending=[True, True])
    .reset_index(drop=True)
)
model_comparison_results.insert(
    0,
    "rank",
    np.arange(1, len(model_comparison_results) + 1),
)
display(model_comparison_results)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 16. Final candidate selection

# COMMAND ----------

best_model_mae = float(model_comparison_results["mae"].min())
near_best_models = model_comparison_results[
    model_comparison_results["mae"]
    <= best_model_mae * (1 + FINAL_MODEL_MAE_TOLERANCE)
].copy()
near_best_models["feature_count"] = near_best_models["feature_list"].apply(len)
near_best_models["model_complexity_rank"] = (
    near_best_models["model_family"]
    .map({"random_forest": 1, "catboost": 2})
    .fillna(999)
)
selected_candidate = near_best_models.sort_values(
    ["feature_count", "model_complexity_rank", "mae"],
    ascending=[True, True, True],
).iloc[0]
display(selected_candidate.to_frame("value"))
print("Candidate selection complete; set UNLOCK_FINAL_TEST = True for final test.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 16b. Production pipeline parity (validation)
# MAGIC
# MAGIC Same feature path as `train.py`: `BusinessBaseline` → `raw_to_feature_frame` →
# MAGIC `build_sklearn_pipeline`. Compare this to the lab's raw-feature candidate.

# COMMAND ----------

training_config = load_training_config()
parity_baseline = BusinessBaseline().fit(train_df)
region_medians = {
    tuple(key.split("|")): value for key, value in parity_baseline.lookup.items()
}
X_train_prod = raw_to_feature_frame(train_df.to_dict("records"), region_medians)
X_validation_prod = raw_to_feature_frame(
    validation_df.to_dict("records"),
    region_medians,
)
production_pipeline = build_sklearn_pipeline(training_config.make_estimator())
production_pipeline.fit(X_train_prod, train_df[TARGET_COLUMN])
production_validation_predictions = production_pipeline.predict(X_validation_prod)

production_validation_metrics = calculate_metrics(
    validation_df[TARGET_COLUMN],
    production_validation_predictions,
)
parity_baseline_metrics = calculate_metrics(
    validation_df[TARGET_COLUMN],
    parity_baseline.predict(validation_df),
)

selected_model_key = selected_candidate["model_key"]
selected_features = list(selected_candidate["feature_list"])
selected_categorical_features = list(selected_candidate["categorical_features"])
if selected_candidate["model_family"] == "random_forest":
    lab_validation_predictions = validation_models[selected_model_key].predict(
        validation_df[selected_features]
    )
else:
    X_validation_cat = prepare_catboost_data(
        validation_df,
        selected_features,
        selected_categorical_features,
    )
    lab_validation_predictions = validation_models[selected_model_key].predict(
        X_validation_cat
    )
lab_validation_metrics = calculate_metrics(
    validation_df[TARGET_COLUMN],
    lab_validation_predictions,
)

production_parity_comparison = pd.DataFrame(
    [
        {"path": "BusinessBaseline", **parity_baseline_metrics},
        {
            "path": (
                f"Lab raw features ({selected_candidate['model_family']}: "
                f"{selected_candidate['feature_set']})"
            ),
            **lab_validation_metrics,
        },
        {
            "path": (
                f"Production engineered features ({training_config.model_type})"
            ),
            **production_validation_metrics,
        },
    ]
)
display(production_parity_comparison)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 17. Final test evaluation

# COMMAND ----------

FINAL_TEST_EVALUATED = UNLOCK_FINAL_TEST
final_test_comparison = pd.DataFrame()
test_mae_improvement_pct = float("nan")

if not FINAL_TEST_EVALUATED:
    print(
        "Final test evaluation is locked. Review validation and production parity, "
        "then set UNLOCK_FINAL_TEST = True and rerun from section 17."
    )
else:
    train_validation_df = pd.concat([train_df, validation_df], axis=0).sort_values(
        DATE_COLUMN
    )
    selected_model_family = selected_candidate["model_family"]

    if selected_model_family == "random_forest":
        numeric, categorical = split_feature_types(selected_features, feature_inventory)
        final_model = build_random_forest_pipeline(
            numeric,
            categorical,
            n_estimators=RF_FINAL_ESTIMATORS,
        )
        final_model.fit(
            train_validation_df[selected_features],
            train_validation_df[TARGET_COLUMN],
        )
        test_predictions = final_model.predict(test_df[selected_features])
    elif selected_model_family == "catboost":
        final_iterations = max(int(selected_candidate["best_iteration"]), 50)
        X_train_validation_cat = prepare_catboost_data(
            train_validation_df,
            selected_features,
            selected_categorical_features,
        )
        X_test_cat = prepare_catboost_data(
            test_df,
            selected_features,
            selected_categorical_features,
        )
        final_model = CatBoostRegressor(
            iterations=final_iterations,
            learning_rate=0.05,
            depth=6,
            loss_function="MAE",
            eval_metric="MAE",
            cat_features=selected_categorical_features,
            l2_leaf_reg=5.0,
            random_strength=1.0,
            random_seed=RANDOM_STATE,
            thread_count=-1,
            verbose=False,
            allow_writing_files=False,
        )
        final_model.fit(
            X_train_validation_cat,
            train_validation_df[TARGET_COLUMN],
            verbose=False,
        )
        test_predictions = final_model.predict(X_test_cat)
    else:
        raise ValueError(f"Unsupported model family: {selected_model_family}")

    test_model_metrics = calculate_metrics(test_df[TARGET_COLUMN], test_predictions)
    test_baseline = BusinessBaseline().fit(train_validation_df)
    test_baseline_predictions = test_baseline.predict(test_df)
    test_baseline_metrics = calculate_metrics(
        test_df[TARGET_COLUMN],
        test_baseline_predictions,
    )
    final_test_comparison = pd.DataFrame(
        [
            {"model": "BusinessBaseline", **test_baseline_metrics},
            {
                "model": (
                    f"{selected_model_family}: {selected_candidate['feature_set']}"
                ),
                **test_model_metrics,
            },
        ]
    )
    display(final_test_comparison)

    test_mae_improvement = test_baseline_metrics["mae"] - test_model_metrics["mae"]
    test_mae_improvement_pct = (
        test_mae_improvement / test_baseline_metrics["mae"] * 100
    )
    print(
        "Final test MAE improvement versus baseline: "
        f"{test_mae_improvement:,.2f} ({test_mae_improvement_pct:.2f}%)"
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## 18. Residual analysis

# COMMAND ----------

residual_results = pd.DataFrame()
shap_importance = pd.DataFrame()

if FINAL_TEST_EVALUATED:
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
        residual_results[TARGET_COLUMN] - residual_results["predicted_price"]
    )
    residual_results["absolute_error"] = residual_results["residual"].abs()
    residual_results["absolute_percentage_error"] = (
        residual_results["absolute_error"] / residual_results[TARGET_COLUMN] * 100
    )
    residual_results["price_bucket"] = pd.qcut(
        residual_results[TARGET_COLUMN],
        q=4,
        duplicates="drop",
    )
    residual_results["surface_bucket"] = pd.qcut(
        residual_results["surface_area"],
        q=4,
        duplicates="drop",
    )

    for segment in [
        "region",
        "property_type",
        "number_of_bedrooms",
        "price_bucket",
        "surface_bucket",
    ]:
        print(f"Residuals by {segment}")
        display(
            residual_segment_summary(
                residual_results,
                segment,
                MIN_RESIDUAL_SEGMENT_SIZE,
            )
        )

    display(
        residual_results.sort_values("absolute_error", ascending=False).head(30)
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
    plt.hist(residual_results["residual"], bins=20)
    plt.axvline(0, linestyle="--")
    plt.xlabel("Residual: actual - predicted")
    plt.ylabel("Count")
    plt.title("Residual distribution")
    plt.tight_layout()
    plt.show()
else:
    print("Residual analysis skipped because final test is locked.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 19. SHAP

# COMMAND ----------

if FINAL_TEST_EVALUATED and RUN_SHAP and shap is not None:
    sample_size = min(200, len(test_df))
    sample = test_df.sample(n=sample_size, random_state=RANDOM_STATE)

    if selected_model_family == "catboost":
        X_shap = prepare_catboost_data(
            sample,
            selected_features,
            selected_categorical_features,
        )
        explainer = shap.TreeExplainer(final_model)
        shap_values = explainer(X_shap)
        shap.plots.bar(shap_values, max_display=20)
        shap.plots.beeswarm(shap_values, max_display=20)
        shap_importance = (
            pd.DataFrame(
                {
                    "feature": X_shap.columns,
                    "mean_absolute_shap": np.abs(shap_values.values).mean(axis=0),
                }
            )
            .sort_values("mean_absolute_shap", ascending=False)
            .reset_index(drop=True)
        )

    elif selected_model_family == "random_forest":
        preprocessor = final_model.named_steps["preprocessor"]
        estimator = final_model.named_steps["model"]
        transformed = preprocessor.transform(sample[selected_features])
        if hasattr(transformed, "toarray"):
            transformed = transformed.toarray()
        feature_names = preprocessor.get_feature_names_out()
        transformed_df = pd.DataFrame(
            transformed,
            columns=feature_names,
            index=sample.index,
        )
        explainer = shap.TreeExplainer(estimator)
        shap_values = explainer(transformed_df, check_additivity=False)
        shap.plots.bar(shap_values, max_display=20)
        shap.plots.beeswarm(shap_values, max_display=20)
        shap_importance = (
            pd.DataFrame(
                {
                    "feature": transformed_df.columns,
                    "mean_absolute_shap": np.abs(shap_values.values).mean(axis=0),
                }
            )
            .sort_values("mean_absolute_shap", ascending=False)
            .reset_index(drop=True)
        )

    display(shap_importance)
elif not FINAL_TEST_EVALUATED:
    print("SHAP skipped because final test is locked.")
else:
    print("SHAP disabled or unavailable.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 20. Assessment gates

# COMMAND ----------

gate_metrics = (
    test_model_metrics
    if FINAL_TEST_EVALUATED
    else lab_validation_metrics
)
gate_baseline_metrics = (
    test_baseline_metrics
    if FINAL_TEST_EVALUATED
    else parity_baseline_metrics
)

assessment_gates = pd.DataFrame(
    [
        {
            "gate": "beats_business_baseline",
            "observed": gate_metrics["mae"] < gate_baseline_metrics["mae"],
            "threshold": True,
            "passed": gate_metrics["mae"] < gate_baseline_metrics["mae"],
        },
        {
            "gate": "minimum_r2",
            "observed": gate_metrics["r2"],
            "threshold": 0.80,
            "passed": gate_metrics["r2"] >= 0.80,
        },
        {
            "gate": "maximum_mape_pct",
            "observed": gate_metrics["mape_pct"],
            "threshold": 15.0,
            "passed": gate_metrics["mape_pct"] <= 15.0,
        },
        {
            "gate": "production_path_not_worse_than_lab",
            "observed": production_validation_metrics["mae"],
            "threshold": lab_validation_metrics["mae"] * 1.05,
            "passed": production_validation_metrics["mae"]
            <= lab_validation_metrics["mae"] * 1.05,
        },
    ]
)
display(assessment_gates)
assessment_passed = bool(assessment_gates["passed"].all())
gate_scope = "final_test" if FINAL_TEST_EVALUATED else "validation"
print(f"Assessment gates passed ({gate_scope}): {assessment_passed}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 21. MLflow logging

# COMMAND ----------

with mlflow.start_run(run_name="assessment_feature_discovery") as run:
    mlflow.set_tags(
        {
            "lane": "assessment",
            "register_model": "false",
            "git_commit": GIT_COMMIT,
            "data_source": DATA_SOURCE,
            "catalog": CATALOG,
            "selected_model_family": selected_candidate["model_family"],
            "selected_feature_set": selected_candidate["feature_set"],
            "final_test_used_once": str(FINAL_TEST_EVALUATED).lower(),
            "assessment_gates_passed": str(assessment_passed),
            "gate_scope": gate_scope,
        }
    )
    mlflow.log_params(
        {
            "train_fraction": TRAIN_FRACTION,
            "validation_fraction": VALIDATION_FRACTION,
            "max_feature_group_depth": MAX_FEATURE_GROUP_DEPTH,
            "rf_search_estimators": RF_SEARCH_ESTIMATORS,
            "rf_comparison_estimators": RF_COMPARISON_ESTIMATORS,
            "rf_final_estimators": RF_FINAL_ESTIMATORS,
            "catboost_enabled": RUN_CATBOOST,
            "production_model_type": training_config.model_type,
        }
    )
    logged_metrics = {
        "validation_lab_mae": lab_validation_metrics["mae"],
        "validation_production_mae": production_validation_metrics["mae"],
        "validation_baseline_mae": parity_baseline_metrics["mae"],
        "assessment_gates_passed": 1.0 if assessment_passed else 0.0,
    }
    if FINAL_TEST_EVALUATED:
        logged_metrics.update(
            {
                "test_mae": test_model_metrics["mae"],
                "test_rmse": test_model_metrics["rmse"],
                "test_r2": test_model_metrics["r2"],
                "test_mape_pct": test_model_metrics["mape_pct"],
                "baseline_mae": test_baseline_metrics["mae"],
                "mae_improvement_pct": test_mae_improvement_pct,
            }
        )
    mlflow.log_metrics(logged_metrics)
    mlflow.log_table(
        mlflow_safe_frame(duplicate_summary),
        "data_quality/duplicate_summary.json",
    )
    mlflow.log_table(
        mlflow_safe_frame(source_cleaning_funnel),
        "data_quality/source_cleaning_funnel.json",
    )
    mlflow.log_table(
        mlflow_safe_frame(model_cleaning_funnel),
        "data_quality/model_cleaning_funnel.json",
    )
    mlflow.log_table(
        mlflow_safe_frame(feature_inventory),
        "eda/feature_inventory.json",
    )
    mlflow.log_table(
        mlflow_safe_frame(numeric_target_report),
        "eda/numeric_target_report.json",
    )
    mlflow.log_table(
        mlflow_safe_frame(single_feature_results),
        "experiments/single_feature_screen.json",
    )
    mlflow.log_table(
        mlflow_safe_frame(feature_matrix_results),
        "experiments/feature_matrix.json",
    )
    mlflow.log_table(
        mlflow_safe_frame(best_by_depth),
        "experiments/best_by_depth.json",
    )
    mlflow.log_table(
        mlflow_safe_frame(group_frequency),
        "experiments/group_frequency.json",
    )
    mlflow.log_table(
        mlflow_safe_frame(model_comparison_results),
        "experiments/model_comparison.json",
    )
    mlflow.log_table(
        mlflow_safe_frame(production_parity_comparison),
        "evaluation/production_parity_validation.json",
    )
    if not final_test_comparison.empty:
        mlflow.log_table(
            mlflow_safe_frame(final_test_comparison),
            "evaluation/final_test.json",
        )
    if not residual_results.empty:
        mlflow.log_table(
            mlflow_safe_frame(residual_results),
            "evaluation/residuals.json",
        )
    mlflow.log_table(
        mlflow_safe_frame(assessment_gates),
        "evaluation/assessment_gates.json",
    )
    if not shap_importance.empty:
        mlflow.log_table(
            mlflow_safe_frame(shap_importance),
            "explainability/shap_importance.json",
        )
    assessment_run_id = run.info.run_id

print(f"MLflow run ID: {assessment_run_id}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 22. Conclusion and production handoff
# MAGIC
# MAGIC The assessment notebook demonstrates the complete experiment process.
# MAGIC Production promotion still requires the official pipeline for walk-forward validation,
# MAGIC production quality gates, train/serve parity, packaging, registration, deployment, and rollback.

# COMMAND ----------

print("Assessment experiment complete.")
print(
    "Selected candidate: "
    f"{selected_candidate['model_family']} — {selected_candidate['feature_set']}"
)
print(
    "Validation MAE (lab / production / baseline): "
    f"{lab_validation_metrics['mae']:,.2f} / "
    f"{production_validation_metrics['mae']:,.2f} / "
    f"{parity_baseline_metrics['mae']:,.2f}"
)
if FINAL_TEST_EVALUATED:
    print(f"Final test MAE: {test_model_metrics['mae']:,.2f}")
    print(f"Improvement versus baseline: {test_mae_improvement_pct:.2f}%")
else:
    print("Final test not evaluated yet. Set UNLOCK_FINAL_TEST = True when ready.")
print("Next step: transfer winning ideas into ml/src and run the official training pipeline.")
