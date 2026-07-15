"""Training YAML configuration loader tests."""

from pathlib import Path

import pytest
import yaml
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge

from house_price_ml.config.training_config import (
    TrainingConfig,
    load_training_config,
    resolve_training_config_path,
)


def test_default_config_path_exists():
    path = resolve_training_config_path()
    assert path.name == "training.yaml"
    assert path.is_file()


def test_load_default_training_config():
    config = load_training_config()
    assert config.model_type == "random_forest"
    assert config.random_forest.n_estimators == 200
    assert config.random_forest.max_depth == 12
    assert config.splits.walk_forward_folds == 3
    assert config.splits.holdout_test_quarters == 2


def test_make_estimator_random_forest():
    config = load_training_config()
    estimator = config.make_estimator()
    assert isinstance(estimator, RandomForestRegressor)
    assert estimator.n_estimators == 200
    assert estimator.max_depth == 12


def test_make_estimator_ridge(tmp_path: Path):
    config_path = tmp_path / "training.yaml"
    config_path.write_text(
        yaml.dump(
            {
                "model_type": "ridge",
                "ridge": {"alpha": 2.5},
                "splits": {"holdout_test_quarters": 1},
            }
        )
    )
    config = load_training_config(config_path)
    estimator = config.make_estimator()
    assert isinstance(estimator, Ridge)
    assert estimator.alpha == 2.5


def test_invalid_config_rejected(tmp_path: Path):
    bad_path = tmp_path / "bad.yaml"
    bad_path.write_text("not_a_mapping")
    with pytest.raises(ValueError, match="YAML mapping"):
        load_training_config(bad_path)


def test_mlflow_params_include_hyperparameters():
    config = TrainingConfig()
    params = config.mlflow_params()
    assert params["rf_n_estimators"] == 200
    assert params["holdout_test_quarters"] == 2
