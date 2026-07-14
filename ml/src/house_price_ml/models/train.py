"""Model training entrypoint."""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from mlflow import MlflowClient
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge

from house_price_ml.config.mlflow_tracking import configure_mlflow
from house_price_ml.config.settings import Settings, get_settings
from house_price_ml.data.silver import bronze_to_silver
from house_price_ml.evaluation.metrics import compute_metrics, evaluate_by_segment
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


def _walk_forward_model_maes(
    splits: list[tuple[pd.DataFrame, pd.DataFrame]],
    model_type: str,
    region_medians_dict: dict[tuple[str, str], float],
) -> list[float]:
    maes: list[float] = []
    estimators = {
        "ridge": Ridge(alpha=1.0),
        "random_forest": RandomForestRegressor(
            n_estimators=200, random_state=42, n_jobs=-1, max_depth=12
        ),
    }
    for train_fold, val_fold in splits:
        estimator = estimators.get(model_type, estimators["random_forest"])
        fold_pipeline = build_sklearn_pipeline(estimator)
        X_tr = raw_to_feature_frame(train_fold.to_dict("records"), region_medians_dict)
        y_tr = train_fold["label_sale_price"].values.astype(float)
        fold_pipeline.fit(X_tr, y_tr)
        X_val = raw_to_feature_frame(val_fold.to_dict("records"), region_medians_dict)
        y_val = val_fold["label_sale_price"].values.astype(float)
        y_hat = fold_pipeline.predict(X_val)
        maes.append(compute_metrics(y_val, y_hat)["mae"])
    return maes


def _feature_importance(pipeline, feature_names: list[str]) -> dict[str, float] | None:
    regressor = pipeline.named_steps.get("regressor")
    if regressor is None or not hasattr(regressor, "feature_importances_"):
        return None
    importances = regressor.feature_importances_
    pairs = sorted(zip(feature_names, importances), key=lambda item: item[1], reverse=True)
    return {name: float(value) for name, value in pairs}


def _log_json_artifact(payload: dict | list, filename: str) -> None:
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
        json.dump(payload, handle, indent=2)
        temp_path = handle.name
    try:
        mlflow.log_artifact(temp_path, artifact_path="reports")
    finally:
        Path(temp_path).unlink(missing_ok=True)


def _register_model_if_requested(
    settings: Settings,
    catalog: str | None,
    model_alias: str | None,
    model_path: Path,
) -> str | None:
    if not catalog or not model_alias:
        return None
    if not settings.databricks_host or not settings.databricks_token:
        return None

    model_name = f"{catalog}.{settings.databricks_schema}.house_price_model"
    active = mlflow.active_run()
    if active is not None:
        model_uri = f"runs:/{active.info.run_id}/model"
    else:
        model_uri = model_path.resolve().as_uri()

    registered = mlflow.register_model(model_uri=model_uri, name=model_name)
    version = str(registered.version)
    client = MlflowClient(registry_uri=mlflow.get_registry_uri())
    client.set_registered_model_alias(model_name, model_alias, version)
    mlflow.log_param("registered_model_name", model_name)
    mlflow.log_param("registered_model_version", version)
    mlflow.log_param("registered_model_alias", model_alias)
    return version


def train(
    data_path: Path,
    model_type: str = "random_forest",
    output_dir: Path | None = None,
    *,
    catalog: str | None = None,
    model_alias: str | None = None,
) -> Path:
    settings = get_settings()
    tracking_uri = configure_mlflow(settings)

    raw = pd.read_csv(data_path, parse_dates=["listing_timestamp", "sale_date"])
    clean, rejected = bronze_to_silver(raw)
    gold = silver_to_gold_features(clean)

    train_df, test_df = holdout_test_split(gold)
    splits = walk_forward_splits(train_df)

    baseline = BusinessBaseline().fit(train_df)
    y_train = train_df["label_sale_price"].values.astype(float)

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
    feature_names = list(X_train.columns)
    pipeline.fit(X_train, y_train)

    wf_baseline_maes = []
    for tr, val in splits:
        bl = BusinessBaseline().fit(tr)
        y_v = val["label_sale_price"].values.astype(float)
        wf_baseline_maes.append(compute_metrics(y_v, bl.predict(val))["mae"])

    wf_model_maes = _walk_forward_model_maes(splits, model_type, region_medians_dict)

    X_test = raw_to_feature_frame(test_df.to_dict("records"), region_medians_dict)
    y_test = test_df["label_sale_price"].values.astype(float)
    y_pred = pipeline.predict(X_test)
    test_metrics = compute_metrics(y_test, y_pred)
    baseline_metrics = compute_metrics(y_test, baseline.predict(test_df))
    beats_baseline = test_metrics["mae"] < baseline_metrics["mae"]
    mae_improvement_pct = (
        round((1 - test_metrics["mae"] / baseline_metrics["mae"]) * 100, 2)
        if baseline_metrics["mae"]
        else 0.0
    )

    holdout = test_df[["listing_id", "region", "property_type", "label_sale_price"]].copy()
    holdout["predicted_price"] = y_pred
    holdout["baseline_price"] = baseline.predict(test_df)
    holdout["residual"] = holdout["label_sale_price"] - holdout["predicted_price"]
    holdout["abs_error"] = holdout["residual"].abs()
    segment_region = evaluate_by_segment(holdout, "label_sale_price", "predicted_price", "region")
    segment_property = evaluate_by_segment(
        holdout, "label_sale_price", "predicted_price", "property_type"
    )

    out = output_dir or Path("artifacts/model")
    out.mkdir(parents=True, exist_ok=True)

    metadata = {
        "feature_pipeline_version": settings.feature_pipeline_version,
        "feature_bounds": get_training_feature_bounds(train_df, region_medians_dict),
        "region_medians": region_medians,
        "model_type": model_type,
        "training_date": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "training_data_rows": len(train_df),
        "validation_approach": "walk_forward + holdout_test",
        "mlflow_tracking_uri": tracking_uri,
        "app_env": settings.app_env,
        "catalog": catalog or settings.databricks_catalog,
    }

    run_name = f"train_{model_type}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    with mlflow.start_run(run_name=run_name):
        mlflow.set_tags(
            {
                "app_env": settings.app_env,
                "model_type": model_type,
                "git_commit": metadata["git_commit"],
                "beats_baseline": str(beats_baseline),
                "data_source": str(data_path),
            }
        )
        mlflow.log_params(
            {
                "model_type": model_type,
                "feature_pipeline_version": settings.feature_pipeline_version,
                "git_commit": metadata["git_commit"],
                "training_rows": len(train_df),
                "test_rows": len(test_df),
                "rejected_rows": len(rejected),
                "walk_forward_folds": len(splits),
                "feature_count": len(feature_names),
                "region_count": train_df["region"].nunique(),
                "property_type_count": train_df["property_type"].nunique(),
                "app_env": settings.app_env,
                "catalog": catalog or settings.databricks_catalog,
                "model_alias": model_alias or settings.model_alias,
                "data_path": str(data_path),
            }
        )

        mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})
        mlflow.log_metrics({f"baseline_{k}": v for k, v in baseline_metrics.items()})
        mlflow.log_metric("beats_baseline", 1.0 if beats_baseline else 0.0)
        mlflow.log_metric("mae_improvement_pct", mae_improvement_pct)
        if wf_baseline_maes:
            mlflow.log_metric("walk_forward_baseline_mae_mean", float(np.mean(wf_baseline_maes)))
            mlflow.log_metric("walk_forward_baseline_mae_std", float(np.std(wf_baseline_maes)))
        if wf_model_maes:
            mlflow.log_metric("walk_forward_model_mae_mean", float(np.mean(wf_model_maes)))
            mlflow.log_metric("walk_forward_model_mae_std", float(np.std(wf_model_maes)))
            mlflow.log_metric(
                "walk_forward_model_beats_baseline",
                1.0 if np.mean(wf_model_maes) < np.mean(wf_baseline_maes) else 0.0,
            )

        model_path = out / "mlflow_model"
        save_model_artifact(pipeline, baseline, metadata, str(model_path))
        mlflow.log_artifacts(str(model_path), artifact_path="model")

        summary = {
            "test_metrics": test_metrics,
            "baseline_metrics": baseline_metrics,
            "beats_baseline": beats_baseline,
            "mae_improvement_pct": mae_improvement_pct,
            "walk_forward_baseline_mae_mean": float(np.mean(wf_baseline_maes)) if wf_baseline_maes else None,
            "walk_forward_model_mae_mean": float(np.mean(wf_model_maes)) if wf_model_maes else None,
        }
        (out / "training_summary.json").write_text(json.dumps(summary, indent=2))

        manifest = _build_training_manifest(
            model_type=model_type,
            metadata=metadata,
            train_df=train_df,
            test_df=test_df,
            baseline=baseline,
            test_metrics=test_metrics,
            baseline_metrics=baseline_metrics,
            summary=summary,
            wf_baseline_maes=wf_baseline_maes,
            wf_model_maes=wf_model_maes,
        )
        _write_training_manifest(
            manifest_dir=Path(__file__).resolve().parents[4] / "netlify" / "functions" / "_shared",
            manifest=manifest,
        )

        _log_json_artifact(summary, "training_summary.json")
        _log_json_artifact(manifest, "training_manifest.json")
        _log_json_artifact(metadata, "model_metadata.json")
        _log_json_artifact(segment_region.to_dict(orient="records"), "metrics_by_region.json")
        _log_json_artifact(segment_property.to_dict(orient="records"), "metrics_by_property_type.json")

        importance = _feature_importance(pipeline, feature_names)
        if importance:
            _log_json_artifact(importance, "feature_importance.json")

        holdout_path = out / "holdout_predictions.csv"
        holdout.to_csv(holdout_path, index=False)
        mlflow.log_artifact(str(holdout_path), artifact_path="reports")

        registered_version = _register_model_if_requested(
            settings, catalog, model_alias, model_path
        )
        if registered_version:
            print(f"Registered {catalog}.{settings.databricks_schema}.house_price_model v{registered_version}")

    return out


def _build_training_manifest(
    model_type: str,
    metadata: dict,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    baseline: BusinessBaseline,
    test_metrics: dict[str, float],
    baseline_metrics: dict[str, float],
    summary: dict,
    wf_baseline_maes: list[float],
    wf_model_maes: list[float],
) -> dict:
    baseline_mae = baseline_metrics["mae"]
    return {
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
        "walk_forward_baseline_mae_mean": float(np.mean(wf_baseline_maes)) if wf_baseline_maes else None,
        "walk_forward_model_mae_mean": float(np.mean(wf_model_maes)) if wf_model_maes else None,
    }


def _write_training_manifest(manifest_dir: Path, manifest: dict) -> None:
    manifest_dir.mkdir(parents=True, exist_ok=True)
    (manifest_dir / "training_manifest.json").write_text(json.dumps(manifest, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model-type", choices=["ridge", "random_forest"], default="random_forest")
    parser.add_argument("--output", type=Path, default=Path("artifacts/model"))
    parser.add_argument("--catalog", default=None, help="Unity Catalog for model registration")
    parser.add_argument("--model-alias", default=None, help="Alias after registration (challenger/champion)")
    args = parser.parse_args()
    path = train(
        args.data,
        args.model_type,
        args.output,
        catalog=args.catalog,
        model_alias=args.model_alias,
    )
    print(f"Model saved to {path}")


if __name__ == "__main__":
    main()
