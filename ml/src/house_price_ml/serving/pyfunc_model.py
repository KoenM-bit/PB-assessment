"""MLflow pyfunc entrypoint (loaded from file path for Databricks Model Serving)."""

from __future__ import annotations

import json

import mlflow
import mlflow.pyfunc
import numpy as np
import pandas as pd

from house_price_ml.features.pipeline import check_out_of_range, compute_row_features
from house_price_ml.models.baseline import BusinessBaseline


class HousePriceModel(mlflow.pyfunc.PythonModel):
    """Accepts raw listing fields; preprocessing runs inside the served artifact."""

    def load_context(self, context: mlflow.pyfunc.PythonModelContext) -> None:
        with open(context.artifacts["metadata"], "r") as f:
            self.metadata = json.load(f)
        self.sklearn_pipeline = mlflow.sklearn.load_model(context.artifacts["sklearn_model"])
        self.baseline = BusinessBaseline.load(context.artifacts["baseline"])
        self.feature_bounds = self.metadata.get("feature_bounds", {})
        self.region_medians = {
            tuple(k.split("|")): v for k, v in self.metadata.get("region_medians", {}).items()
        }

    def predict(
        self,
        context: mlflow.pyfunc.PythonModelContext,
        model_input: pd.DataFrame,
    ) -> pd.DataFrame:
        rows = model_input.to_dict("records")
        predictions = []
        all_warnings: list[list[str]] = []

        for row in rows:
            features = compute_row_features(row, self.region_medians)
            warnings = check_out_of_range(features, self.feature_bounds)
            feature_df = pd.DataFrame([features])
            try:
                pred = float(self.sklearn_pipeline.predict(feature_df)[0])
                if not np.isfinite(pred) or pred <= 0:
                    pred = self.baseline.predict_row(row)
                    warnings.append("fallback_to_baseline_invalid_prediction")
            except Exception:
                pred = self.baseline.predict_row(row)
                warnings.append("fallback_to_baseline_on_error")
            predictions.append(pred)
            all_warnings.append(warnings)

        return pd.DataFrame(
            {
                "predicted_price": predictions,
                "warnings": [json.dumps(w) for w in all_warnings],
            }
        )


mlflow.models.set_model(HousePriceModel())
