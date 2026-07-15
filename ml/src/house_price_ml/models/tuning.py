"""Hyperparameter tuning via walk-forward validation."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np
import pandas as pd

from house_price_ml.config.training_config import TrainingConfig
from house_price_ml.evaluation.metrics import compute_metrics
from house_price_ml.evaluation.splits import walk_forward_splits
from house_price_ml.features.pipeline import raw_to_feature_frame
from house_price_ml.models.baseline import BusinessBaseline
from house_price_ml.serving.mlflow_model import build_sklearn_pipeline


def _region_medians_dict_from_df(df: pd.DataFrame) -> dict[tuple[str, str], float]:
    baseline = BusinessBaseline().fit(df)
    return {tuple(k.split("|")): v for k, v in baseline.lookup.items()}


def _mean_walk_forward_mae(train_df: pd.DataFrame, config: TrainingConfig) -> float:
    splits = walk_forward_splits(
        train_df,
        n_splits=config.splits.walk_forward_folds,
        test_size_quarters=config.splits.walk_forward_test_quarters,
    )
    if not splits:
        return float("inf")

    maes: list[float] = []
    for train_fold, val_fold in splits:
        fold_medians = _region_medians_dict_from_df(train_fold)
        pipeline = build_sklearn_pipeline(config.make_estimator())
        X_tr = raw_to_feature_frame(train_fold.to_dict("records"), fold_medians)
        y_tr = train_fold["label_sale_price"].values.astype(float)
        pipeline.fit(X_tr, y_tr)
        X_val = raw_to_feature_frame(val_fold.to_dict("records"), fold_medians)
        y_val = val_fold["label_sale_price"].values.astype(float)
        y_hat = pipeline.predict(X_val)
        maes.append(compute_metrics(y_val, y_hat)["mae"])
    return float(np.mean(maes))


def tune_hyperparameters(train_df: pd.DataFrame, config: TrainingConfig) -> tuple[TrainingConfig, dict[str, Any]]:
    """Return config with best hyperparameters and tuning metadata."""
    if not config.tuning.enabled:
        return config, {"tuning_skipped": True}

    if config.tuning.method == "grid":
        return _grid_search(train_df, config)

    try:
        import optuna
    except ImportError as exc:
        raise ImportError("Install tuning extras: pip install -e 'ml/[tuning]'") from exc

    study = optuna.create_study(direction="minimize")

    def objective(trial: optuna.Trial) -> float:
        trial_config = deepcopy(config)
        if config.model_type == "random_forest":
            trial_config.random_forest.n_estimators = trial.suggest_categorical(
                "n_estimators", config.search_space.random_forest.n_estimators
            )
            depth = trial.suggest_categorical("max_depth", config.search_space.random_forest.max_depth)
            trial_config.random_forest.max_depth = depth if depth is not None else 32
        else:
            trial_config.ridge.alpha = trial.suggest_categorical(
                "alpha", config.search_space.ridge.alpha
            )
        return _mean_walk_forward_mae(train_df, trial_config)

    study.optimize(objective, n_trials=config.tuning.n_trials, show_progress_bar=False)
    best = study.best_params
    tuned = deepcopy(config)
    if config.model_type == "random_forest":
        tuned.random_forest.n_estimators = int(best["n_estimators"])
        tuned.random_forest.max_depth = int(best["max_depth"])
    else:
        tuned.ridge.alpha = float(best["alpha"])

    meta = {
        "tuning_method": "optuna",
        "n_trials": config.tuning.n_trials,
        "best_value": study.best_value,
        "best_params": best,
    }
    return tuned, meta


def _grid_search(train_df: pd.DataFrame, config: TrainingConfig) -> tuple[TrainingConfig, dict[str, Any]]:
    best_mae = float("inf")
    best_config = config
    best_params: dict[str, Any] = {}

    if config.model_type == "random_forest":
        for n_est in config.search_space.random_forest.n_estimators:
            for depth in config.search_space.random_forest.max_depth:
                trial = deepcopy(config)
                trial.random_forest.n_estimators = n_est
                trial.random_forest.max_depth = depth if depth is not None else 32
                mae = _mean_walk_forward_mae(train_df, trial)
                if mae < best_mae:
                    best_mae = mae
                    best_config = trial
                    best_params = {"n_estimators": n_est, "max_depth": depth}
    else:
        for alpha in config.search_space.ridge.alpha:
            trial = deepcopy(config)
            trial.ridge.alpha = alpha
            mae = _mean_walk_forward_mae(train_df, trial)
            if mae < best_mae:
                best_mae = mae
                best_config = trial
                best_params = {"alpha": alpha}

    return best_config, {
        "tuning_method": "grid",
        "best_value": best_mae,
        "best_params": best_params,
    }
