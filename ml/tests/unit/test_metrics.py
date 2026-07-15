"""Tests for evaluation metrics."""

from __future__ import annotations

import numpy as np

from house_price_ml.evaluation.metrics import (
    compute_metrics,
    mdape,
    pct_within_tolerance,
)


def test_pct_within_tolerance():
    y_true = np.array([100.0, 200.0, 300.0, 400.0])
    y_pred = np.array([105.0, 180.0, 350.0, 500.0])
    assert pct_within_tolerance(y_true, y_pred, 0.10) == 0.5


def test_mdape_robust_to_outlier():
    y_true = np.array([100.0, 100.0, 100.0, 1000.0])
    y_pred = np.array([110.0, 105.0, 95.0, 200.0])
    assert mdape(y_true, y_pred) < 20.0


def test_compute_metrics_includes_pct_within():
    y_true = np.array([200000.0, 300000.0, 400000.0])
    y_pred = np.array([210000.0, 290000.0, 450000.0])
    metrics = compute_metrics(y_true, y_pred)
    assert "pct_within_10pct" in metrics
    assert "mdape" in metrics
