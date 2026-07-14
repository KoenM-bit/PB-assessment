"""Drift detection interfaces."""

from __future__ import annotations

from typing import Protocol

import numpy as np
import pandas as pd


class DriftCalculator(Protocol):
    def score(self, reference: pd.Series, recent: pd.Series) -> float: ...


class PercentOutOfRangeCalculator:
    """Simple drift proxy: percentage of recent values outside reference p01-p99."""

    def score(self, reference: pd.Series, recent: pd.Series) -> float:
        if reference.empty or recent.empty:
            return 0.0
        p01, p99 = reference.quantile(0.01), reference.quantile(0.99)
        outside = (recent < p01) | (recent > p99)
        return float(outside.mean() * 100)


class MeanShiftCalculator:
    """Normalised mean shift between reference and recent."""

    def score(self, reference: pd.Series, recent: pd.Series) -> float:
        if reference.empty or recent.empty:
            return 0.0
        ref_std = reference.std()
        if ref_std == 0:
            return 0.0
        return float(abs(recent.mean() - reference.mean()) / ref_std)


def compute_feature_monitoring(
    reference: pd.DataFrame,
    recent: pd.DataFrame,
    feature_cols: list[str],
    drift_calc: DriftCalculator | None = None,
) -> pd.DataFrame:
    calc = drift_calc or PercentOutOfRangeCalculator()
    rows = []
    for col in feature_cols:
        if col not in reference.columns or col not in recent.columns:
            continue
        ref = reference[col].dropna()
        rec = recent[col].dropna()
        rows.append(
            {
                "feature_name": col,
                "reference_mean": float(ref.mean()) if len(ref) else 0.0,
                "reference_std": float(ref.std()) if len(ref) else 0.0,
                "recent_mean": float(rec.mean()) if len(rec) else 0.0,
                "recent_std": float(rec.std()) if len(rec) else 0.0,
                "pct_out_of_range": calc.score(ref, rec),
                "drift_score": calc.score(ref, rec),
                "sample_size": len(rec),
            }
        )
    return pd.DataFrame(rows)
