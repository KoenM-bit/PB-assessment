"""Model training entrypoint."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge

from house_price_ml.config.settings import get_settings
from house_price_ml.data.silver import bronze_to_silver
from house_price_ml.evaluation.metrics import compute_metrics
from house_price_ml.evaluation.splits import holdout_test_split, walk_forward_splits
from house_price_ml.features.aggregates import silver_to_gold_features
from house_price_ml.features.pipeline import get_training_feature_bounds, raw_to_feature_frame
from house_price_ml.models.baseline import BusinessBaseline
from house_price_ml.serving.mlflow_model import build_sklearn_pipeline, save_model_artifact


def _git_commit() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def train(data_path: Path, model_type: str = "random_forest", output_dir: Path | None = None) -> Path:
    settings = get_settings()
    raw = pd.read_csv(data_path, parse_dates=["listing_timestamp", "sale_date"])
    clean, _ = bronze_to_silver(raw)
    gold = silver_to_gold_features(clean)

    train_df, test_df = holdout_test_split(gold)
    splits = walk_forward_splits(train_df)

    baseline = BusinessBaseline().fit(train_df)
    y_train = train_df["label_sale_price"].values.astype(float)

    # Region medians for serving
    region_medians = {
        f"{r}|{p}": baseline.lookup.get(f"{r}|{p}", baseline.global_median_psm)
        for r in train_df["region"].unique()
        for p in train_df["property_type"].unique()
    }

    estimators = {
        "ridge": Ridge(alpha=1.0),
        "random_forest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    }
    estimator = estimators.get(model_type, estimators["random_forest"])
    if isinstance(estimator, RandomForestRegressor):
        estimator = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1, max_depth=12)
    pipeline = build_sklearn_pipeline(estimator)

    region_medians_dict = {tuple(k.split("|")): v for k, v in region_medians.items()}
    X_train = raw_to_feature_frame(train_df.to_dict("records"), region_medians_dict)
    pipeline.fit(X_train, y_train)

    # Walk-forward validation metrics
    wf_maes = []
    for tr, val in splits:
        bl = BusinessBaseline().fit(tr)
        X_v = raw_to_feature_frame(val.to_dict("records"))
        y_v = val["label_sale_price"].values.astype(float)
        wf_maes.append(compute_metrics(y_v, bl.predict(val))["mae"])

    X_test = raw_to_feature_frame(test_df.to_dict("records"), region_medians_dict)
    y_test = test_df["label_sale_price"].values.astype(float)
    y_pred = pipeline.predict(X_test)
    test_metrics = compute_metrics(y_test, y_pred)
    baseline_metrics = compute_metrics(y_test, baseline.predict(test_df))

    out = output_dir or Path("artifacts/model")
    out.mkdir(parents=True, exist_ok=True)

    metadata = {
        "feature_pipeline_version": settings.feature_pipeline_version,
        "feature_bounds": get_training_feature_bounds(train_df, region_medians_dict),
        "region_medians": region_medians,
        "model_type": model_type,
        "training_date": datetime.utcnow().isoformat(),
        "git_commit": _git_commit(),
        "training_data_rows": len(train_df),
        "validation_approach": "walk_forward + holdout_test",
    }

    mlflow.set_experiment("house_price_prediction")
    with mlflow.start_run(run_name=f"train_{model_type}"):
        mlflow.log_params(
            {
                "model_type": model_type,
                "feature_pipeline_version": settings.feature_pipeline_version,
                "git_commit": metadata["git_commit"],
                "training_rows": len(train_df),
                "test_rows": len(test_df),
            }
        )
        mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})
        mlflow.log_metrics({f"baseline_{k}": v for k, v in baseline_metrics.items()})
        if wf_maes:
            mlflow.log_metric("walk_forward_baseline_mae_mean", float(np.mean(wf_maes)))

        model_path = out / "mlflow_model"
        save_model_artifact(pipeline, baseline, metadata, str(model_path))
        mlflow.log_artifacts(str(model_path), artifact_path="model")

        summary = {
            "test_metrics": test_metrics,
            "baseline_metrics": baseline_metrics,
            "beats_baseline": test_metrics["mae"] < baseline_metrics["mae"],
        }
        (out / "training_summary.json").write_text(json.dumps(summary, indent=2))
        _write_training_manifest(
            manifest_dir=Path(__file__).resolve().parents[4] / "netlify" / "functions" / "_shared",
            model_type=model_type,
            metadata=metadata,
            train_df=train_df,
            test_df=test_df,
            baseline=baseline,
            test_metrics=test_metrics,
            baseline_metrics=baseline_metrics,
            summary=summary,
            wf_maes=wf_maes,
        )

    return out


def _write_training_manifest(
    manifest_dir: Path,
    model_type: str,
    metadata: dict,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    baseline: BusinessBaseline,
    test_metrics: dict[str, float],
    baseline_metrics: dict[str, float],
    summary: dict,
    wf_maes: list[float],
) -> None:
    baseline_mae = baseline_metrics["mae"]
    manifest = {
        "model_type": model_type,
        "feature_pipeline_version": metadata["feature_pipeline_version"],
        "training_date": metadata["training_date"],
        "git_commit": metadata["git_commit"],
        "training_data_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "validation_approach": metadata["validation_approach"],
        "regions": sorted(train_df["region"].unique().tolist()),
        "property_types": sorted(train_df["property_type"].unique().tolist()),
        "surface_area_range": {
            "min": float(train_df["surface_area"].min()),
            "max": float(train_df["surface_area"].max()),
            "median": float(train_df["surface_area"].median()),
        },
        "price_range": {
            "min": float(train_df["label_sale_price"].min()),
            "max": float(train_df["label_sale_price"].max()),
            "median": float(train_df["label_sale_price"].median()),
        },
        "feature_bounds": metadata["feature_bounds"],
        "baseline_lookup": baseline.lookup,
        "global_median_psm": baseline.global_median_psm,
        "holdout_evaluation": {
            "model": {**test_metrics, "sample_size": int(len(test_df))},
            "baseline": {**baseline_metrics, "sample_size": int(len(test_df))},
            "beats_baseline": summary["beats_baseline"],
            "mae_improvement_pct": round((1 - test_metrics["mae"] / baseline_mae) * 100, 1)
            if baseline_mae
            else 0.0,
        },
        "walk_forward_baseline_mae_mean": float(np.mean(wf_maes)) if wf_maes else None,
    }
    manifest_dir.mkdir(parents=True, exist_ok=True)
    (manifest_dir / "training_manifest.json").write_text(json.dumps(manifest, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model-type", choices=["ridge", "random_forest"], default="random_forest")
    parser.add_argument("--output", type=Path, default=Path("artifacts/model"))
    args = parser.parse_args()
    path = train(args.data, args.model_type, args.output)
    print(f"Model saved to {path}")


if __name__ == "__main__":
    main()
