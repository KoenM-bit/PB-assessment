"""Environment-aware application settings."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


def _resolve_env_files() -> tuple[str, ...]:
    """Find repo-root .env even when cwd is ml/ (e.g. make train)."""
    paths: list[str] = []
    for parent in Path(__file__).resolve().parents:
        candidate = parent / ".env"
        if candidate.is_file():
            paths.append(str(candidate))
    cwd_env = Path.cwd() / ".env"
    if cwd_env.is_file() and str(cwd_env) not in paths:
        paths.append(str(cwd_env))
    return tuple(paths) if paths else (".env",)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=_resolve_env_files(), extra="ignore")

    app_env: Literal["local", "staging", "production"] = "local"
    databricks_host: str = ""
    databricks_token: str = ""
    databricks_serving_endpoint: str = "house-price-serving"
    databricks_sql_warehouse_id: str = ""
    databricks_catalog: str = "house_price_staging"
    databricks_schema: str = "gold"
    model_alias: Literal["champion", "challenger", "previous_champion"] = "challenger"
    min_evaluation_sample_size: int = 30
    demo_write_token: str = ""
    serving_timeout_ms: int = 10000
    use_mock_databricks: bool = True
    feature_pipeline_version: str = "1.0.0"
    mlflow_tracking_uri: str = ""
    mlflow_registry_uri: str = ""
    mlflow_experiment_name: str = "/Shared/house_price_prediction"

    @property
    def bronze_schema(self) -> str:
        return "bronze"

    @property
    def silver_schema(self) -> str:
        return "silver"

    @property
    def gold_schema(self) -> str:
        return "gold"

    def full_table(self, schema: str, table: str) -> str:
        return f"{self.databricks_catalog}.{schema}.{table}"


@lru_cache
def get_settings() -> Settings:
    return Settings()
