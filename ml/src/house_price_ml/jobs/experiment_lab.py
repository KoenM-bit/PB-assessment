"""Reusable helpers for the ML experiment lab notebook (testable, minimal Spark)."""

from __future__ import annotations

import os
from typing import Any

import pandas as pd
from mlflow import MlflowClient

from house_price_ml.config.mlflow_tracking import configure_mlflow
from house_price_ml.config.settings import get_settings
from house_price_ml.data.data_config import load_data_profile
from house_price_ml.data.silver import bronze_to_silver
from house_price_ml.data.synthetic import generate_listings
from house_price_ml.data.training_data import assemble_training_frame
from house_price_ml.evaluation.gates import GateResult
from house_price_ml.evaluation.metrics import compute_metrics
from house_price_ml.evaluation.segments import price_category
from house_price_ml.features.aggregates import silver_to_gold_features
from house_price_ml.models.baseline import BusinessBaseline

LAB_MLFLOW_EXPERIMENT = "/Shared/house_price_prediction_lab"

_KEY_NULL_COLUMNS = (
    "region",
    "property_type",
    "surface_area",
    "label_sale_price",
    "sale_date",
)


def configure_lab_mlflow(experiment_name: str = LAB_MLFLOW_EXPERIMENT) -> str:
    """Point MLflow at the lab experiment (no UC registration in this lane)."""
    os.environ["MLFLOW_EXPERIMENT_NAME"] = experiment_name
    settings = get_settings()
    return configure_mlflow(settings, experiment_name=experiment_name)


def data_quality_summary(
    silver_df: pd.DataFrame,
    rejected_df: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Row counts, reject rate, and null rates on key training columns."""
    silver_rows = len(silver_df)
    rejected_rows = len(rejected_df) if rejected_df is not None else 0
    bronze_rows = silver_rows + rejected_rows
    reject_rate = round(rejected_rows / bronze_rows, 4) if bronze_rows else 0.0

    null_rates: dict[str, float] = {}
    for col in _KEY_NULL_COLUMNS:
        if col in silver_df.columns and silver_rows:
            null_rates[col] = round(float(silver_df[col].isna().mean()), 4)

    return {
        "bronze_rows": bronze_rows,
        "silver_rows": silver_rows,
        "rejected_rows": rejected_rows,
        "reject_rate": reject_rate,
        "null_rates": null_rates,
    }


def segment_balance_report(training_df: pd.DataFrame) -> pd.DataFrame:
    """Counts by region, property_type, and price_category for segment exploration."""
    frames: list[pd.DataFrame] = []
    for segment_col in ("region", "property_type"):
        if segment_col not in training_df.columns:
            continue
        counts = training_df[segment_col].value_counts().reset_index()
        counts.columns = ["segment", "count"]
        counts.insert(0, "segment_type", segment_col)
        frames.append(counts)

    if "label_sale_price" in training_df.columns:
        price_df = training_df.copy()
        price_df["price_category"] = price_df["label_sale_price"].apply(price_category)
        counts = price_df["price_category"].value_counts().reset_index()
        counts.columns = ["segment", "count"]
        counts.insert(0, "segment_type", "price_category")
        frames.append(counts)

    if not frames:
        return pd.DataFrame(columns=["segment_type", "segment", "count"])
    return pd.concat(frames, ignore_index=True)


def baseline_holdout_metrics(train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict[str, float]:
    """Business baseline MAE on a holdout slice (no MLflow logging)."""
    baseline = BusinessBaseline().fit(train_df)
    y_test = test_df["label_sale_price"].values.astype(float)
    preds = baseline.predict(test_df)
    metrics = compute_metrics(y_test, preds)
    return {"baseline_mae": float(metrics["mae"]), "baseline_rmse": float(metrics["rmse"])}


def format_gate_report(gate_result: GateResult) -> pd.DataFrame:
    """Flatten gate violations into a display-friendly table."""
    rows: list[dict[str, Any]] = []
    for failure in gate_result.failures:
        rows.append({"check": failure, "segment_type": "", "segment": "", "detail": ""})

    for key, value in gate_result.details.items():
        if not key.endswith("_violations") or not isinstance(value, list):
            continue
        segment_type = key.replace("_violations", "")
        for item in value:
            rows.append(
                {
                    "check": f"{segment_type} MAE degradation",
                    "segment_type": segment_type,
                    "segment": item.get("segment", ""),
                    "detail": (
                        f"mae={item.get('mae')} ratio={item.get('ratio_vs_overall')} "
                        f"n={item.get('sample_size')}"
                    ),
                }
            )

    if not rows:
        return pd.DataFrame(
            [{"check": "all gates passed", "segment_type": "", "segment": "", "detail": ""}]
        )
    return pd.DataFrame(rows)


def search_recent_runs(experiment_name: str, n: int = 10) -> pd.DataFrame:
    """Compare recent runs (mae, gates, git commit, training lane)."""
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        return pd.DataFrame(
            columns=[
                "run_id",
                "run_name",
                "start_time",
                "test_mae",
                "gates_passed",
                "beats_baseline",
                "git_commit",
                "training_lane",
            ]
        )

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"],
        max_results=n,
    )
    rows: list[dict[str, Any]] = []
    for run in runs:
        metrics = run.data.metrics
        tags = run.data.tags
        rows.append(
            {
                "run_id": run.info.run_id,
                "run_name": run.info.run_name,
                "start_time": run.info.start_time,
                "test_mae": metrics.get("test_mae"),
                "gates_passed": metrics.get("gates_passed"),
                "beats_baseline": tags.get("beats_baseline"),
                "git_commit": tags.get("git_commit"),
                "training_lane": tags.get("training_lane"),
            }
        )
    return pd.DataFrame(rows)


def load_training_frame_from_catalog(spark: Any, catalog: str) -> pd.DataFrame:
    """Load assembled training frame from silver + gold Delta tables."""
    silver_table = f"{catalog}.silver.listings_clean"
    gold_table = f"{catalog}.gold.listing_features"
    silver_df = spark.table(silver_table).toPandas()
    gold_df = spark.table(gold_table).toPandas()
    return assemble_training_frame(silver_df, gold_df)


def load_training_frame_sample(profile: str | None = None, rows: int | None = None) -> pd.DataFrame:
    """In-memory training frame via synthetic bronze → silver → gold (no Delta writes)."""
    prof = load_data_profile(profile)
    n = rows or prof.rows
    raw = generate_listings(
        n,
        seed=prof.seed,
        missing_rate=prof.missing_rate,
        outlier_rate=prof.outlier_rate,
        invalid_rate=prof.invalid_rate,
        start_year=prof.start_year,
        span_days=prof.span_days,
    )
    clean, _rejected = bronze_to_silver(raw)
    gold = silver_to_gold_features(clean)
    return assemble_training_frame(clean, gold)
