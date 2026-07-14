"""Build monitoring metric records for Gold tables."""

from __future__ import annotations

from datetime import date
import uuid

import pandas as pd

from house_price_ml.evaluation.metrics import compute_metrics, evaluate_by_segment
from house_price_ml.evaluation.segments import add_evaluation_segments, price_category


def build_retrospective_evaluations(
    joined: pd.DataFrame,
    model_version: str,
    window_type: str = "all_time",
) -> pd.DataFrame:
    """Build model evaluation records from joined predictions and actuals."""
    if joined.empty:
        return pd.DataFrame()

    joined = joined.copy()
    joined["actual_sale_price"] = joined["actual_sale_price"].astype(float)
    joined["predicted_price"] = joined["predicted_price"].astype(float)
    joined = add_evaluation_segments(joined)

    rows: list[dict] = []
    eval_date = date.today()

    overall = compute_metrics(
        joined["actual_sale_price"].values,
        joined["predicted_price"].values,
    )
    rows.append(
        _eval_row(eval_date, window_type, "overall", "all", len(joined), overall, model_version)
    )

    for segment_type, col in [
        ("region", "region"),
        ("property_type", "property_type"),
        ("price_category", "price_category"),
    ]:
        seg_df = evaluate_by_segment(joined, "actual_sale_price", "predicted_price", col)
        for _, seg in seg_df.iterrows():
            rows.append(
                _eval_row(
                    eval_date,
                    window_type,
                    segment_type,
                    str(seg["segment"]),
                    int(seg["sample_size"]),
                    {k: seg[k] for k in ["mae", "rmse", "bias", "mape"]},
                    model_version,
                )
            )

    return pd.DataFrame(rows)


def _eval_row(
    eval_date: date,
    window_type: str,
    segment_type: str,
    segment_value: str,
    sample_size: int,
    metrics: dict,
    model_version: str,
) -> dict:
    return {
        "evaluation_id": str(uuid.uuid4()),
        "evaluation_date": eval_date,
        "window_type": window_type,
        "segment_type": segment_type,
        "segment_value": segment_value,
        "sample_size": sample_size,
        "mae": metrics["mae"],
        "rmse": metrics["rmse"],
        "bias": metrics["bias"],
        "mape": metrics["mape"],
        "model_version": model_version,
    }


def assign_price_category_from_prediction(predicted_price: float) -> str:
    return price_category(predicted_price)
