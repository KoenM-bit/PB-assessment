"""UC registration opt-in defaults for train()."""

from __future__ import annotations

import pytest

from house_price_ml.config.settings import Settings
from house_price_ml.models.train import _resolve_register_model


@pytest.fixture
def databricks_settings() -> Settings:
    return Settings(
        databricks_host="https://example.databricks.com",
        databricks_token="token",
    )


def test_register_default_false_on_databricks(monkeypatch, databricks_settings):
    monkeypatch.setenv("DATABRICKS_RUNTIME_VERSION", "14.3")
    monkeypatch.delenv("REGISTER_UC_MODEL", raising=False)
    assert _resolve_register_model(None, databricks_settings) is False


def test_register_explicit_true(monkeypatch, databricks_settings):
    monkeypatch.setenv("DATABRICKS_RUNTIME_VERSION", "14.3")
    assert _resolve_register_model(True, databricks_settings) is True


def test_register_explicit_false(monkeypatch, databricks_settings):
    monkeypatch.setenv("DATABRICKS_RUNTIME_VERSION", "14.3")
    assert _resolve_register_model(False, databricks_settings) is False


def test_register_env_opt_in(monkeypatch, databricks_settings):
    monkeypatch.setenv("DATABRICKS_RUNTIME_VERSION", "14.3")
    monkeypatch.setenv("REGISTER_UC_MODEL", "true")
    assert _resolve_register_model(None, databricks_settings) is True


def test_register_without_credentials():
    offline = Settings(databricks_host="", databricks_token="")
    assert _resolve_register_model(True, offline) is False
