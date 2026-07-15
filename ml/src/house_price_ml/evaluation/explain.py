"""Model explainability helpers."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def compute_shap_summary(
    pipeline,
    X_sample: pd.DataFrame,
    *,
    max_features: int = 15,
) -> dict[str, Any]:
    """Compute mean |SHAP| for top features on a holdout sample."""
    try:
        import shap
    except ImportError:
        return {"enabled": False, "error": "shap not installed — pip install -e 'ml/[analysis]'"}

    preprocessor = pipeline.named_steps.get("preprocessor")
    regressor = pipeline.named_steps.get("regressor")
    if preprocessor is None or regressor is None:
        return {"enabled": False, "error": "pipeline missing preprocessor or regressor"}

    X_transformed = preprocessor.transform(X_sample)
    if hasattr(preprocessor, "get_feature_names_out"):
        feature_names = list(preprocessor.get_feature_names_out())
    else:
        feature_names = [f"feature_{i}" for i in range(X_transformed.shape[1])]

    if hasattr(regressor, "estimators_"):
        explainer = shap.TreeExplainer(regressor)
    else:
        explainer = shap.Explainer(regressor, X_transformed)

    shap_values = explainer.shap_values(X_transformed)
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    mean_abs = np.abs(shap_values).mean(axis=0)
    top_idx = np.argsort(mean_abs)[::-1][:max_features]

    return {
        "enabled": True,
        "sample_size": int(len(X_sample)),
        "top_features": [
            {
                "feature": str(feature_names[i]),
                "mean_abs_shap": round(float(mean_abs[i]), 4),
            }
            for i in top_idx
        ],
    }
