"""Model training entrypoint."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from mlflow import MlflowClient
from sklearn.base import BaseEstimator

from house_price_ml.config.mlflow_tracking import configure_mlflow
from house_price_ml.config.settings import Settings, get_settings
from house_price_ml.config.training_config import (
    TrainingConfig,
    load_training_config,
    resolve_training_config_path,
)
from house_price_ml.data.training_data import (
    TrainingDataSource,
    load_export_metadata,
    load_training_frame,
)
from house_price_ml.evaluation.metrics import compute_metrics, evaluate_by_segment
from house_price_ml.evaluation.splits import holdout_test_split, walk_forward_splits
from house_price_ml.features.pipeline import get_training_feature_bounds, raw_to_feature_frame
from house_price_ml.models.baseline import BusinessBaseline
from house_price_ml.serving.mlflow_model import build_sklearn_pipeline, save_model_artifact


def _git_commit(explicit: str | None = None) -> str:
    if explicit and explicit not in ("unknown", "none", ""):
        return explicit
    for key in ("GIT_COMMIT", "GITHUB_SHA"):
        value = os.environ.get(key, "").strip()
        if value and value not in ("unknown", "none"):
            return value
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def _training_source() -> str:
    if os.environ.get("DATABRICKS_RUNTIME_VERSION"):
        return "databricks"
    if os.environ.get("GITHUB_ACTIONS") == "true":
        return "github_actions"
    return "local"


def _can_register_to_uc(settings: Settings) -> bool:
    if os.environ.get("DATABRICKS_RUNTIME_VERSION"):
        return True
    return bool(settings.databricks_host and settings.databricks_token)


def _resolve_register_model(register_model: bool | None, settings: Settings) -> bool:
    """Default False — only register when explicitly opted in (pipeline / make train --register)."""
    if register_model is False:
        return False
    if not _can_register_to_uc(settings):
        return False
    if register_model is True:
        return True
    if os.environ.get("REGISTER_UC_MODEL", "").strip().lower() in {"1", "true", "yes"}:
        return True
    return False


def _make_estimator(config: TrainingConfig) -> BaseEstimator:
    return config.make_estimator()


def _region_medians_dict_from_df(df: pd.DataFrame) -> dict[tuple[str, str], float]:
    """Region×property_type median price/sqm lookup from a dataframe slice."""
    baseline = BusinessBaseline().fit(df)
    return {tuple(k.split("|")): v for k, v in baseline.lookup.items()}


def _walk_forward_model_maes(
    splits: list[tuple[pd.DataFrame, pd.DataFrame]],
    config: TrainingConfig,
) -> list[float]:
    maes: list[float] = []
    for train_fold, val_fold in splits:
        # Recompute region medians from the fold train slice only (no future leakage).
        fold_medians = _region_medians_dict_from_df(train_fold)
        fold_pipeline = build_sklearn_pipeline(_make_estimator(config))
        X_tr = raw_to_feature_frame(train_fold.to_dict("records"), fold_medians)
        y_tr = train_fold["label_sale_price"].values.astype(float)
        fold_pipeline.fit(X_tr, y_tr)
        X_val = raw_to_feature_frame(val_fold.to_dict("records"), fold_medians)
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


def _register_model(
    settings: Settings,
    catalog: str | None,
    model_path: Path,
    register_alias: str | None,
) -> str | None:
    if not catalog:
        return None
    if not _can_register_to_uc(settings):
        return None

    model_name = f"{catalog}.{settings.databricks_schema}.house_price_model"
    active = mlflow.active_run()
    if active is not None:
        model_uri = f"runs:/{active.info.run_id}/model"
    else:
        model_uri = model_path.resolve().as_uri()

    registered = mlflow.register_model(model_uri=model_uri, name=model_name)
    version = str(registered.version)
    mlflow.log_param("registered_model_name", model_name)
    mlflow.log_param("registered_model_version", version)

    if register_alias:
        client = MlflowClient(registry_uri=mlflow.get_registry_uri())
        client.set_registered_model_alias(model_name, register_alias, version)
        mlflow.log_param("registered_model_alias", register_alias)

    return version


def train(
    data: TrainingDataSource,
    model_type: str | None = None,
    output_dir: Path | None = None,
    *,
    training_config: TrainingConfig | None = None,
    config_path: Path | str | None = None,
    catalog: str | None = None,
    register_model: bool | None = None,
    register_alias: str | None = None,
    git_commit: str | None = None,
    data_source: str | None = None,
) -> Path:
    settings = get_settings()
    resolved_config_path = resolve_training_config_path(config_path)
    config = training_config or load_training_config(resolved_config_path)
    if model_type is not None:
        config = config.model_copy(update={"model_type": model_type})  # type: ignore[arg-type]
    resolved_model_type = config.model_type
    log_catalog = catalog or settings.databricks_catalog
    register_model = _resolve_register_model(register_model, settings)
    resolved_git_commit = _git_commit(git_commit)
    data_path_str = str(data) if isinstance(data, Path) else "dataframe"
    resolved_data_source = data_source or data_path_str
    tracking_uri = configure_mlflow(settings)

    training_frame = load_training_frame(data)
    rejected_rows = 0
    if isinstance(data, Path):
        export_meta = load_export_metadata(data)
        if export_meta is not None:
            rejected_rows = int(export_meta.get("rejected_rows", 0))

    train_df, test_df = holdout_test_split(
        training_frame,
        test_quarters=config.splits.holdout_test_quarters,
    )
    splits = walk_forward_splits(
        train_df,
        n_splits=config.splits.walk_forward_folds,
        test_size_quarters=config.splits.walk_forward_test_quarters,
    )

    baseline = BusinessBaseline().fit(train_df)
    y_train = train_df["label_sale_price"].values.astype(float)

    region_medians = {
        f"{r}|{p}": baseline.lookup.get(f"{r}|{p}", baseline.global_median_psm)
        for r in train_df["region"].unique()
        for p in train_df["property_type"].unique()
    }

    pipeline = build_sklearn_pipeline(_make_estimator(config))

    region_medians_dict = _region_medians_dict_from_df(train_df)
    X_train = raw_to_feature_frame(train_df.to_dict("records"), region_medians_dict)
    feature_names = list(X_train.columns)
    pipeline.fit(X_train, y_train)

    wf_baseline_maes = []
    for tr, val in splits:
        bl = BusinessBaseline().fit(tr)
        y_v = val["label_sale_price"].values.astype(float)
        wf_baseline_maes.append(compute_metrics(y_v, bl.predict(val))["mae"])

    wf_model_maes = _walk_forward_model_maes(splits, config)

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
        "model_type": resolved_model_type,
        "training_config_path": str(resolved_config_path),
        "training_date": datetime.now(timezone.utc).isoformat(),
        "git_commit": resolved_git_commit,
        "training_data_rows": len(train_df),
        "validation_approach": "walk_forward + holdout_test",
        "mlflow_tracking_uri": tracking_uri,
        "app_env": settings.app_env,
        "catalog": log_catalog,
    }

    run_name = f"train_{resolved_model_type}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    with mlflow.start_run(run_name=run_name):
        mlflow.set_tags(
            {
                "app_env": settings.app_env,
                "model_type": resolved_model_type,
                "git_commit": resolved_git_commit,
                "training_source": _training_source(),
                "beats_baseline": str(beats_baseline),
                "data_source": resolved_data_source,
                "training_config_path": str(resolved_config_path),
            }
        )
        mlflow.log_params(
            {
                "model_type": resolved_model_type,
                "training_config_path": str(resolved_config_path),
                **config.mlflow_params(),
                "feature_pipeline_version": settings.feature_pipeline_version,
                "git_commit": resolved_git_commit,
                "training_rows": len(train_df),
                "test_rows": len(test_df),
                "rejected_rows": rejected_rows,
                "walk_forward_folds": len(splits),
                "feature_count": len(feature_names),
                "region_count": train_df["region"].nunique(),
                "property_type_count": train_df["property_type"].nunique(),
                "app_env": settings.app_env,
                "catalog": log_catalog,
                "data_path": data_path_str,
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
            model_type=resolved_model_type,
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

        registered_version = None
        if register_model:
            registered_version = _register_model(
                settings, log_catalog, model_path, register_alias
            )
        if registered_version:
            if register_alias:
                print(
                    f"Registered {log_catalog}.{settings.databricks_schema}.house_price_model "
                    f"v{registered_version} as @{register_alias}"
                )
            else:
                print(
                    f"Registered {log_catalog}.{settings.databricks_schema}.house_price_model "
                    f"v{registered_version} (no alias — not live in staging)"
                )
                print("Promote when ready: make promote-challenger RUN_ID=<run-id>")
        else:
            print("Experiment logged (model not registered to Unity Catalog).")

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
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Training YAML config (default: ml/config/training.yaml)",
    )
    parser.add_argument(
        "--model-type",
        choices=["ridge", "random_forest"],
        default=None,
        help="Override model_type from training config",
    )
    parser.add_argument("--output", type=Path, default=Path("artifacts/model"))
    parser.add_argument("--catalog", default=None, help="Unity Catalog name (logging only)")
    parser.add_argument(
        "--register-alias",
        default=None,
        help="Also set this UC alias (e.g. challenger). Default: register version only.",
    )
    parser.add_argument(
        "--register",
        action="store_true",
        help="Register model version to Unity Catalog (off by default; pipeline uses this).",
    )
    parser.add_argument(
        "--no-register",
        action="store_true",
        help="Skip Unity Catalog registration (default without --register).",
    )
    parser.add_argument("--git-commit", default=None, help="Git SHA for experiment tags")
    parser.add_argument("--data-source", default=None, help="Human-readable data source label")
    args = parser.parse_args()
    settings = get_settings()
    if args.no_register and args.register:
        parser.error("Use only one of --register or --no-register")
    register_model = True if args.register else False if args.no_register else None
    path = train(
        args.data,
        args.model_type,
        args.output,
        config_path=args.config,
        catalog=args.catalog or settings.databricks_catalog,
        register_model=register_model,
        register_alias=args.register_alias,
        git_commit=args.git_commit,
        data_source=args.data_source,
    )
    print(f"Model saved to {path}")


if __name__ == "__main__":
    main()
