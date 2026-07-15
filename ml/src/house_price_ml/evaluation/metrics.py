"""Regression evaluation metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def bias(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(y_pred - y_true))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = y_true != 0
    if not mask.any():
        return 0.0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def mdape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = y_true != 0
    if not mask.any():
        return 0.0
    ape = np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])
    return float(np.median(ape) * 100)


def pct_within_tolerance(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    tolerance: float = 0.10,
) -> float:
    """Fraction of predictions within +/- tolerance of actual (e.g. 0.10 = 10%)."""
    mask = y_true != 0
    if not mask.any():
        return 0.0
    y_t = y_true[mask]
    y_p = y_pred[mask]
    within = np.abs(y_p - y_t) / y_t <= tolerance
    return float(np.mean(within))


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "bias": bias(y_true, y_pred),
        "mape": mape(y_true, y_pred),
        "mdape": mdape(y_true, y_pred),
        "pct_within_10pct": pct_within_tolerance(y_true, y_pred, 0.10),
    }


def evaluate_by_segment(
    df: pd.DataFrame,
    y_true_col: str,
    y_pred_col: str,
    segment_col: str,
) -> pd.DataFrame:
    rows = []
    for segment, group in df.groupby(segment_col):
        if len(group) == 0:
            continue
        metrics = compute_metrics(
            group[y_true_col].values.astype(float),
            group[y_pred_col].values.astype(float),
        )
        rows.append({"segment": segment, "sample_size": len(group), **metrics})
    return pd.DataFrame(rows)
