"""Training frame loading — single path from gold (no rebuild inside train)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from house_price_ml.data.silver import bronze_to_silver
from house_price_ml.features.aggregates import silver_to_gold_features

# Silver raw fields + gold label/snapshot (model features recomputed in train via raw_to_feature_frame).
TRAINING_FRAME_COLUMNS = [
    "listing_id",
    "listing_timestamp",
    "sale_date",
    "region",
    "postcode",
    "latitude",
    "longitude",
    "surface_area",
    "number_of_rooms",
    "number_of_bedrooms",
    "build_year",
    "energy_label",
    "property_type",
    "garden",
    "feature_snapshot_date",
    "label_sale_price",
]

GOLD_JOIN_COLUMNS = ["listing_id", "feature_snapshot_date", "label_sale_price"]

RAW_BRONZE_INDICATOR_COLUMNS = frozenset(
    {"asking_price", "ingestion_timestamp", "ingestion_date", "source_file"}
)

TrainingDataSource = Path | pd.DataFrame

_EXPORT_META_SUFFIX = ".meta.json"


class TrainingFrameError(ValueError):
    """Raised when training input is not a valid assembled training frame."""


def is_raw_bronze_input(df: pd.DataFrame) -> bool:
    """True when the frame looks like raw bronze/listings CSV, not a training export."""
    if "label_sale_price" in df.columns and "feature_snapshot_date" in df.columns:
        return False
    if RAW_BRONZE_INDICATOR_COLUMNS.intersection(df.columns):
        return True
    return "label_sale_price" not in df.columns


def assemble_training_frame(silver_df: pd.DataFrame, gold_df: pd.DataFrame) -> pd.DataFrame:
    """Join silver clean rows with gold label and snapshot (training read surface)."""
    silver_required = [
        c for c in TRAINING_FRAME_COLUMNS if c not in ("feature_snapshot_date", "label_sale_price")
    ]
    missing_silver = [c for c in silver_required if c not in silver_df.columns]
    if missing_silver:
        raise TrainingFrameError(f"Silver frame missing columns: {missing_silver}")

    gold_cols = [c for c in GOLD_JOIN_COLUMNS if c in gold_df.columns]
    if "listing_id" not in gold_cols or "label_sale_price" not in gold_cols:
        raise TrainingFrameError("Gold frame must include listing_id and label_sale_price")

    gold_subset = gold_df[gold_cols].drop_duplicates(subset=["listing_id"], keep="first")
    merged = silver_df.merge(gold_subset, on="listing_id", how="inner", suffixes=("", "_gold"))

    # Prefer gold label/snapshot when duplicate column names exist after merge.
    if "label_sale_price_gold" in merged.columns:
        merged["label_sale_price"] = merged["label_sale_price_gold"]
        merged = merged.drop(columns=["label_sale_price_gold"])
    if "feature_snapshot_date_gold" in merged.columns:
        merged["feature_snapshot_date"] = merged["feature_snapshot_date_gold"]
        merged = merged.drop(columns=["feature_snapshot_date_gold"])

    return validate_training_frame(merged)


def validate_training_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure required columns exist and rows are usable for training."""
    if df.empty:
        raise TrainingFrameError("Training frame is empty")

    if is_raw_bronze_input(df):
        raise TrainingFrameError(
            "Raw listings.csv is not a training frame. Run: make gold-export"
        )

    missing = [c for c in TRAINING_FRAME_COLUMNS if c not in df.columns]
    if missing:
        raise TrainingFrameError(f"Training frame missing required columns: {missing}")

    result = df[TRAINING_FRAME_COLUMNS].copy()
    result["sale_date"] = pd.to_datetime(result["sale_date"])
    result["listing_timestamp"] = pd.to_datetime(result["listing_timestamp"])
    result["feature_snapshot_date"] = pd.to_datetime(result["feature_snapshot_date"])
    result = result.dropna(subset=["label_sale_price", "sale_date"])
    if result.empty:
        raise TrainingFrameError("No rows with label_sale_price and sale_date after validation")
    return result


def load_training_frame(source: TrainingDataSource) -> pd.DataFrame:
    """Load and validate a pre-built training frame from path or DataFrame."""
    if isinstance(source, pd.DataFrame):
        return validate_training_frame(source)

    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"Training data not found: {path}")

    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        header = pd.read_csv(path, nrows=0)
        date_cols = [
            c
            for c in ("listing_timestamp", "sale_date", "feature_snapshot_date")
            if c in header.columns
        ]
        df = pd.read_csv(path, parse_dates=date_cols or None)

    return validate_training_frame(df)


def export_metadata_path(parquet_path: Path) -> Path:
    """Sidecar metadata path for a training export (rejected_rows, lineage)."""
    stem = parquet_path.name
    if stem.endswith(".parquet"):
        stem = stem[: -len(".parquet")]
    return parquet_path.with_name(f"{stem}{_EXPORT_META_SUFFIX}")


def load_export_metadata(data_path: Path) -> dict[str, Any] | None:
    """Load export metadata written by build_training_export, if present."""
    meta_path = export_metadata_path(Path(data_path))
    if not meta_path.is_file():
        return None
    payload = json.loads(meta_path.read_text())
    return dict(payload)


def build_training_export(
    bronze_path: Path,
    output_path: Path,
    *,
    rejected_rows: int | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    ETL-only: bronze CSV → silver → gold → assembled training frame.

    Used by ``make gold-export`` and mirrors Databricks 02_silver + 03_gold + assemble.
    """
    raw = pd.read_csv(bronze_path, parse_dates=["listing_timestamp", "sale_date"])
    clean, rejected = bronze_to_silver(raw)
    gold = silver_to_gold_features(clean)
    frame = assemble_training_frame(clean, gold)

    metadata: dict[str, Any] = {
        "export_date": datetime.now(timezone.utc).isoformat(),
        "source_file": str(bronze_path.resolve()),
        "output_file": str(output_path.resolve()),
        "training_rows": len(frame),
        "rejected_rows": rejected_rows if rejected_rows is not None else len(rejected),
        "bronze_rows": len(raw),
        "silver_rows": len(clean),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output_path, index=False)
    export_metadata_path(output_path).write_text(json.dumps(metadata, indent=2))

    return frame, metadata


def write_training_export(frame: pd.DataFrame, output_path: Path, metadata: dict[str, Any]) -> None:
    """Persist an already-assembled training frame + metadata."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output_path, index=False)
    export_metadata_path(output_path).write_text(json.dumps(metadata, indent=2))
