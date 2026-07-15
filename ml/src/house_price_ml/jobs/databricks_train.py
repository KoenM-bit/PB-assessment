"""Shared helpers for Databricks train notebooks and jobs."""

from __future__ import annotations

from house_price_ml.config.training_config import TrainingConfig, load_training_config


def parse_bool(value: str | None, *, default: bool = False) -> bool:
    if value is None or value == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def apply_experiment_flags(
    config: TrainingConfig,
    *,
    enable_tuning: bool | None = None,
    enable_ablation: bool | None = None,
    enable_explainability: bool | None = None,
) -> TrainingConfig:
    """Override tuning/ablation/explainability flags (e.g. from job widgets)."""
    updated = config
    if enable_tuning is not None:
        updated = updated.model_copy(
            update={"tuning": updated.tuning.model_copy(update={"enabled": enable_tuning})}
        )
    if enable_ablation is not None:
        updated = updated.model_copy(
            update={"ablation": updated.ablation.model_copy(update={"enabled": enable_ablation})}
        )
    if enable_explainability is not None:
        updated = updated.model_copy(
            update={
                "explainability": updated.explainability.model_copy(
                    update={"enabled": enable_explainability}
                )
            }
        )
    return updated


def training_config_from_job_params(
    *,
    config_path: str | None = None,
    model_type: str | None = None,
    enable_tuning: str | None = None,
    enable_ablation: str | None = None,
    enable_explainability: str | None = None,
) -> TrainingConfig:
    """Build TrainingConfig from Databricks job/notebook widget values."""
    config = load_training_config(config_path)
    if model_type and model_type not in ("", "none"):
        config = config.model_copy(update={"model_type": model_type})  # type: ignore[arg-type]
    return apply_experiment_flags(
        config,
        enable_tuning=parse_bool(enable_tuning) if enable_tuning is not None else None,
        enable_ablation=parse_bool(enable_ablation) if enable_ablation is not None else None,
        enable_explainability=parse_bool(enable_explainability)
        if enable_explainability is not None
        else None,
    )
