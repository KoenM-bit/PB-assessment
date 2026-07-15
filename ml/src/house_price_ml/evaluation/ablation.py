"""Feature-group ablation studies."""

from __future__ import annotations

import pandas as pd

from house_price_ml.config.training_config import TrainingConfig
from house_price_ml.evaluation.metrics import compute_metrics
from house_price_ml.features.pipeline import FEATURE_GROUPS, raw_to_feature_frame
from house_price_ml.models.baseline import BusinessBaseline
from house_price_ml.serving.mlflow_model import build_sklearn_pipeline


def _neutralize_columns(X: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = X.copy()
    for col in columns:
        if col not in out.columns:
            continue
        if pd.api.types.is_numeric_dtype(out[col]):
            out[col] = out[col].median()
        else:
            mode = out[col].mode()
            out[col] = mode.iloc[0] if not mode.empty else out[col].iloc[0]
    return out


def run_ablation(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config: TrainingConfig,
) -> pd.DataFrame:
    """Train full model and ablated variants; return MAE delta per dropped group."""
    baseline = BusinessBaseline().fit(train_df)
    region_medians = {
        tuple(k.split("|")): v for k, v in baseline.lookup.items()
    }
    X_train = raw_to_feature_frame(train_df.to_dict("records"), region_medians)
    X_test = raw_to_feature_frame(test_df.to_dict("records"), region_medians)
    y_train = train_df["label_sale_price"].values.astype(float)
    y_test = test_df["label_sale_price"].values.astype(float)

    full_pipeline = build_sklearn_pipeline(config.make_estimator())
    full_pipeline.fit(X_train, y_train)
    full_mae = compute_metrics(y_test, full_pipeline.predict(X_test))["mae"]

    rows: list[dict] = [{"feature_group": "full", "mae": full_mae, "delta_mae_vs_full": 0.0}]
    for group, columns in FEATURE_GROUPS.items():
        X_tr = _neutralize_columns(X_train, columns)
        X_te = _neutralize_columns(X_test, columns)
        pipeline = build_sklearn_pipeline(config.make_estimator())
        pipeline.fit(X_tr, y_train)
        mae = compute_metrics(y_test, pipeline.predict(X_te))["mae"]
        rows.append(
            {
                "feature_group": group,
                "mae": mae,
                "delta_mae_vs_full": round(mae - full_mae, 2),
            }
        )
    return pd.DataFrame(rows)
