"""Shared pytest fixtures for repo-root tests."""

from __future__ import annotations

import os

import pytest

from house_price_ml.config.settings import get_settings

_OFFLINE_ENV_KEYS = ("DATABRICKS_HOST", "DATABRICKS_TOKEN", "MLFLOW_TRACKING_URI")


@pytest.fixture(scope="module", autouse=True)
def offline_mlflow() -> None:
    saved = {key: os.environ.get(key) for key in _OFFLINE_ENV_KEYS}
    os.environ["DATABRICKS_HOST"] = ""
    os.environ["DATABRICKS_TOKEN"] = ""
    os.environ["MLFLOW_TRACKING_URI"] = "sqlite:///:memory:"
    get_settings.cache_clear()
    yield
    for key, value in saved.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value
    get_settings.cache_clear()
