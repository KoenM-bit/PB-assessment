"""Auto-generated model card for audit and stakeholder review."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from house_price_ml.evaluation.gates import GateResult


def build_model_card(
    *,
    model_type: str,
    metadata: dict[str, Any],
    summary: dict[str, Any],
    gate_result: GateResult,
    feature_names: list[str],
    segment_region: list[dict],
    segment_property: list[dict],
    segment_price: list[dict] | None,
    mlflow_run_id: str | None,
    gates_config_path: str,
    training_config_path: str,
    tuning_meta: dict[str, Any] | None = None,
    shap_summary: dict[str, Any] | None = None,
    ablation_report: list[dict] | None = None,
) -> dict[str, Any]:
    """Build model_card.json payload."""
    return {
        "schema_version": "1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model_description": {
            "name": "house_price_model",
            "type": model_type,
            "intended_use": "Predict Dutch residential sale prices from listing attributes.",
            "limitations": [
                "Trained on synthetic listing data; not validated on live market transactions.",
                "Segment performance may vary by region and property type.",
            ],
        },
        "training_data": {
            "source": metadata.get("data_source", metadata.get("catalog", "unknown")),
            "rows": metadata.get("training_data_rows"),
            "table_versions": metadata.get("training_table_versions"),
            "training_date": metadata.get("training_date"),
            "git_commit": metadata.get("git_commit"),
        },
        "features": {
            "feature_pipeline_version": metadata.get("feature_pipeline_version"),
            "feature_count": len(feature_names),
            "feature_names": feature_names,
        },
        "metrics": {
            "holdout": summary.get("test_metrics"),
            "baseline_holdout": summary.get("baseline_metrics"),
            "beats_baseline": summary.get("beats_baseline"),
            "walk_forward_model_mae_mean": summary.get("walk_forward_model_mae_mean"),
            "walk_forward_baseline_mae_mean": summary.get("walk_forward_baseline_mae_mean"),
            "segments": {
                "region": segment_region,
                "property_type": segment_property,
                "price_category": segment_price or [],
            },
        },
        "quality_gates": {
            "config_path": gates_config_path,
            "passed": gate_result.passed,
            "failures": gate_result.failures,
            "details": gate_result.details,
        },
        "experimentation": {
            "tuning": tuning_meta,
            "ablation": ablation_report,
            "shap": shap_summary,
        },
        "ethical_considerations": [
            "Monitor segment MAE variance across regions and price categories.",
            "Do not use for automated lending or discrimination-sensitive decisions without review.",
        ],
        "lineage": {
            "mlflow_run_id": mlflow_run_id,
            "training_config_path": training_config_path,
            "gates_config_path": gates_config_path,
            "git_commit": metadata.get("git_commit"),
        },
    }
