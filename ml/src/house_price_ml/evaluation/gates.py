"""Training quality gate evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from house_price_ml.evaluation.gate_config import QualityGatesConfig, load_quality_gates


class TrainingGateError(Exception):
    """Raised when training quality gates fail and enforcement is on."""


@dataclass
class GateResult:
    passed: bool
    failures: list[str] = field(default_factory=list)
    details: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "failures": self.failures,
            "details": self.details,
        }


def _check_segment_mae(
    segment_df: pd.DataFrame,
    overall_mae: float,
    segment_type: str,
    config: QualityGatesConfig,
    result: GateResult,
) -> None:
    if segment_df.empty or overall_mae <= 0:
        return
    min_n = config.segments.min_segment_sample_size
    max_ratio = config.segments.max_mae_degradation_vs_overall
    violations: list[dict] = []
    for _, row in segment_df.iterrows():
        if int(row["sample_size"]) < min_n:
            continue
        ratio = float(row["mae"]) / overall_mae
        if ratio > max_ratio:
            violations.append(
                {
                    "segment_type": segment_type,
                    "segment": row["segment"],
                    "mae": float(row["mae"]),
                    "ratio_vs_overall": round(ratio, 3),
                    "sample_size": int(row["sample_size"]),
                }
            )
    if violations:
        result.failures.append(
            f"{segment_type}: {len(violations)} segment(s) exceed MAE degradation threshold"
        )
        result.details[f"{segment_type}_violations"] = violations


def evaluate_training_gates(
    summary: dict,
    segment_region: pd.DataFrame,
    segment_property: pd.DataFrame,
    segment_price: pd.DataFrame | None = None,
    gates: QualityGatesConfig | None = None,
) -> GateResult:
    """Evaluate holdout, walk-forward, and segment gates."""
    gates = gates or load_quality_gates()
    result = GateResult(passed=True)

    test_metrics = summary.get("test_metrics") or {}
    baseline_metrics = summary.get("baseline_metrics") or {}
    test_mae = float(test_metrics.get("mae", 0))
    baseline_mae = float(baseline_metrics.get("mae", 0))
    beats_baseline = bool(summary.get("beats_baseline", False))
    pct_within = float(test_metrics.get("pct_within_10pct", 0))

    if gates.holdout.beats_baseline and not beats_baseline:
        result.failures.append("holdout: model does not beat business baseline on MAE")

    if baseline_mae > 0 and test_mae > baseline_mae * gates.holdout.max_mae_vs_baseline_ratio:
        result.failures.append(
            f"holdout: model MAE {test_mae:.0f} exceeds "
            f"{gates.holdout.max_mae_vs_baseline_ratio:.0%} of baseline MAE"
        )

    if pct_within < gates.holdout.min_pct_within_10pct:
        result.failures.append(
            f"holdout: pct_within_10pct {pct_within:.2%} below "
            f"{gates.holdout.min_pct_within_10pct:.2%}"
        )

    wf_model = summary.get("walk_forward_model_mae_mean")
    wf_baseline = summary.get("walk_forward_baseline_mae_mean")
    if gates.walk_forward.model_beats_baseline and wf_model is not None and wf_baseline is not None:
        if float(wf_model) >= float(wf_baseline):
            result.failures.append("walk_forward: model MAE not better than baseline mean")

    result.details["holdout"] = {
        "test_mae": test_mae,
        "baseline_mae": baseline_mae,
        "beats_baseline": beats_baseline,
        "pct_within_10pct": pct_within,
    }
    result.details["walk_forward"] = {
        "model_mae_mean": wf_model,
        "baseline_mae_mean": wf_baseline,
    }

    _check_segment_mae(segment_region, test_mae, "region", gates, result)
    _check_segment_mae(segment_property, test_mae, "property_type", gates, result)
    if segment_price is not None:
        _check_segment_mae(segment_price, test_mae, "price_category", gates, result)

    result.passed = len(result.failures) == 0
    return result
