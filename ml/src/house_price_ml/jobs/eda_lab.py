"""EDA and feature-engineering helpers for the experiment lab notebook."""

from __future__ import annotations

from typing import Any

import pandas as pd

from house_price_ml.config.eda_lab_config import EdaLabConfig, load_eda_lab_config
from house_price_ml.config.training_config import TrainingConfig, load_training_config
from house_price_ml.evaluation.ablation import _neutralize_columns
from house_price_ml.evaluation.explain import compute_shap_summary
from house_price_ml.evaluation.metrics import compute_metrics, evaluate_by_segment
from house_price_ml.features.pipeline import FEATURE_GROUPS, raw_to_feature_frame
from house_price_ml.models.baseline import BusinessBaseline
from house_price_ml.serving.mlflow_model import build_sklearn_pipeline

_TARGET = "label_sale_price"


def _subsample(df: pd.DataFrame, max_rows: int) -> pd.DataFrame:
    if len(df) <= max_rows:
        return df
    return df.sample(n=max_rows, random_state=42)


def _region_medians_from_df(df: pd.DataFrame) -> dict[tuple[str, str], float]:
    baseline = BusinessBaseline().fit(df)
    return {tuple(k.split("|")): v for k, v in baseline.lookup.items()}


def _feature_frame(df: pd.DataFrame, region_medians: dict[tuple[str, str], float]) -> pd.DataFrame:
    return raw_to_feature_frame(df.to_dict("records"), region_medians)


def _skew(series: pd.Series) -> float:
    clean = series.dropna()
    if len(clean) < 3 or clean.std() == 0:
        return 0.0
    return float(clean.skew())


def _iqr_outlier_rate(series: pd.Series, multiplier: float) -> float:
    clean = series.dropna()
    if clean.empty:
        return 0.0
    q1, q3 = clean.quantile(0.25), clean.quantile(0.75)
    iqr = q3 - q1
    if iqr == 0:
        return 0.0
    lower = q1 - multiplier * iqr
    upper = q3 + multiplier * iqr
    return float(((clean < lower) | (clean > upper)).mean())


def evaluate_data_quality_gates(
    dq_summary: dict[str, Any],
    config: EdaLabConfig,
) -> pd.DataFrame:
    """Turn data_quality_summary output into pass/fail checks vs eda_lab thresholds."""
    rows: list[dict[str, Any]] = []
    null_rates = dq_summary.get("null_rates") or {}
    for col in config.data_quality.key_columns:
        rate = float(null_rates.get(col, 0.0))
        rows.append(
            {
                "check": f"null_rate_{col}",
                "value": rate,
                "threshold": config.data_quality.max_null_rate,
                "passed": rate <= config.data_quality.max_null_rate,
            }
        )
    reject_rate = float(dq_summary.get("reject_rate", 0.0))
    rows.append(
        {
            "check": "reject_rate",
            "value": reject_rate,
            "threshold": config.data_quality.max_reject_rate,
            "passed": reject_rate <= config.data_quality.max_reject_rate,
        }
    )
    return pd.DataFrame(rows)


def univariate_profile(df: pd.DataFrame, config: EdaLabConfig) -> pd.DataFrame:
    """Summary stats, skew, and outlier rate per column."""
    rows: list[dict[str, Any]] = []
    uni = config.univariate

    for col in uni.numeric_columns:
        if col not in df.columns:
            continue
        series = pd.to_numeric(df[col], errors="coerce")
        rows.append(
            {
                "column": col,
                "dtype": "numeric",
                "count": int(series.notna().sum()),
                "null_rate": round(float(series.isna().mean()), 4),
                "mean": round(float(series.mean()), 2) if series.notna().any() else None,
                "std": round(float(series.std()), 2) if series.notna().any() else None,
                "p25": round(float(series.quantile(0.25)), 2) if series.notna().any() else None,
                "p50": round(float(series.quantile(0.50)), 2) if series.notna().any() else None,
                "p75": round(float(series.quantile(0.75)), 2) if series.notna().any() else None,
                "skew": round(_skew(series), 3),
                "skew_warning": abs(_skew(series)) >= uni.skew_warning_threshold,
                "outlier_rate": round(_iqr_outlier_rate(series, uni.outlier_iqr_multiplier), 4),
            }
        )

    for col in uni.categorical_columns:
        if col not in df.columns:
            continue
        series = df[col]
        nunique = int(series.nunique(dropna=True))
        mode = series.mode()
        rows.append(
            {
                "column": col,
                "dtype": "categorical",
                "count": int(series.notna().sum()),
                "null_rate": round(float(series.isna().mean()), 4),
                "n_unique": nunique,
                "top_category": str(mode.iloc[0]) if not mode.empty else None,
                "top_category_pct": round(float((series == mode.iloc[0]).mean()), 4)
                if not mode.empty
                else None,
                "skew": None,
                "skew_warning": False,
                "outlier_rate": None,
            }
        )
    return pd.DataFrame(rows)


def bivariate_vs_target(df: pd.DataFrame, config: EdaLabConfig) -> pd.DataFrame:
    """Numeric correlations and categorical mean deltas vs target."""
    if _TARGET not in df.columns:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    target = pd.to_numeric(df[_TARGET], errors="coerce")
    overall_mean = float(target.mean())

    for col in config.bivariate.numeric_vs_target:
        if col not in df.columns:
            continue
        numeric = pd.to_numeric(df[col], errors="coerce")
        valid = numeric.notna() & target.notna()
        if valid.sum() < 3:
            continue
        corr = float(numeric[valid].corr(target[valid]))
        rows.append(
            {
                "feature": col,
                "type": "numeric",
                "correlation_with_target": round(corr, 4),
                "segment": None,
                "segment_mean_target": None,
                "delta_vs_overall_mean": None,
            }
        )

    for col in config.bivariate.categorical_vs_target:
        if col not in df.columns:
            continue
        grouped = df.groupby(col)[_TARGET].mean()
        for segment, mean_val in grouped.items():
            rows.append(
                {
                    "feature": col,
                    "type": "categorical",
                    "correlation_with_target": None,
                    "segment": str(segment),
                    "segment_mean_target": round(float(mean_val), 2),
                    "delta_vs_overall_mean": round(float(mean_val - overall_mean), 2),
                }
            )
    return pd.DataFrame(rows)


def correlation_report(df: pd.DataFrame, config: EdaLabConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Full correlation matrix and flagged pairs (multicollinearity + weak target signal)."""
    cols = [c for c in config.correlation.numeric_columns if c in df.columns]
    if len(cols) < 2:
        return pd.DataFrame(), pd.DataFrame()

    numeric = df[cols].apply(pd.to_numeric, errors="coerce")
    method = config.correlation.method
    matrix = numeric.corr(method=method).round(4)

    pairs: list[dict[str, Any]] = []
    for i, col_a in enumerate(cols):
        for col_b in cols[i + 1 :]:
            value = float(matrix.loc[col_a, col_b])
            if abs(value) >= config.correlation.high_correlation_threshold:
                pairs.append(
                    {
                        "feature_a": col_a,
                        "feature_b": col_b,
                        "correlation": round(value, 4),
                        "flag": "high_multicollinearity",
                    }
                )

    if _TARGET in cols:
        for col in cols:
            if col == _TARGET:
                continue
            value = float(matrix.loc[col, _TARGET])
            if abs(value) < config.correlation.min_abs_correlation_with_target:
                pairs.append(
                    {
                        "feature_a": col,
                        "feature_b": _TARGET,
                        "correlation": round(value, 4),
                        "flag": "weak_target_signal",
                    }
                )

    return matrix, pd.DataFrame(pairs)


def evaluate_business_hypotheses(df: pd.DataFrame, config: EdaLabConfig) -> pd.DataFrame:
    """Run configurable business sanity checks."""
    rows: list[dict[str, Any]] = []
    target_series = pd.to_numeric(df[_TARGET], errors="coerce") if _TARGET in df.columns else None
    overall_mean = float(target_series.mean()) if target_series is not None else 0.0

    for hyp in config.business_hypotheses:
        passed = False
        observed: float | None = None
        threshold: float | None = None

        if hyp.type == "correlation" and hyp.x and hyp.y and hyp.x in df.columns and hyp.y in df.columns:
            x = pd.to_numeric(df[hyp.x], errors="coerce")
            y = pd.to_numeric(df[hyp.y], errors="coerce")
            valid = x.notna() & y.notna()
            observed = float(x[valid].corr(y[valid])) if valid.sum() >= 3 else 0.0
            threshold = hyp.min_correlation
            passed = observed >= (threshold or 0.0)

        elif (
            hyp.type == "segment_mean_ratio"
            and hyp.segment_column
            and hyp.segment_value
            and hyp.target
            and hyp.segment_column in df.columns
            and hyp.target in df.columns
        ):
            seg_mean = float(df.loc[df[hyp.segment_column] == hyp.segment_value, hyp.target].mean())
            observed = seg_mean / overall_mean if overall_mean else 0.0
            threshold = hyp.min_ratio_vs_overall
            passed = observed >= (threshold or 0.0)

        rows.append(
            {
                "hypothesis": hyp.name,
                "type": hyp.type,
                "observed": round(observed, 4) if observed is not None else None,
                "threshold": threshold,
                "passed": passed,
            }
        )
    return pd.DataFrame(rows)


def _columns_for_groups(include_groups: list[str]) -> list[str]:
    columns: list[str] = []
    for group in include_groups:
        columns.extend(FEATURE_GROUPS.get(group, []))
    return list(dict.fromkeys(columns))


def _drop_groups_columns(
    X: pd.DataFrame,
    include_groups: list[str],
) -> pd.DataFrame:
    keep_columns = set(_columns_for_groups(include_groups))
    all_group_cols: list[str] = []
    for cols in FEATURE_GROUPS.values():
        all_group_cols.extend(cols)
    to_neutralize = [c for c in all_group_cols if c not in keep_columns and c in X.columns]
    return _neutralize_columns(X, to_neutralize)


def run_feature_matrix(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config: EdaLabConfig,
    training_config: TrainingConfig | None = None,
) -> pd.DataFrame:
    """Matrix-style feature group experiments from eda_lab.yaml."""
    fm = config.feature_matrix
    if not fm.enabled or not fm.experiments:
        return pd.DataFrame()

    train_slice = _subsample(train_df, fm.max_train_rows)
    tcfg = training_config or load_training_config()
    tcfg = tcfg.model_copy(update={"model_type": fm.model_type})

    baseline = BusinessBaseline().fit(train_slice)
    region_medians = {tuple(k.split("|")): v for k, v in baseline.lookup.items()}
    X_train_full = _feature_frame(train_slice, region_medians)
    X_test_full = _feature_frame(test_df, region_medians)
    y_train = train_slice[_TARGET].values.astype(float)
    y_test = test_df[_TARGET].values.astype(float)

    full_pipeline = build_sklearn_pipeline(tcfg.make_estimator())
    full_pipeline.fit(X_train_full, y_train)
    full_mae = compute_metrics(y_test, full_pipeline.predict(X_test_full))["mae"]

    rows: list[dict[str, Any]] = []
    for exp in fm.experiments:
        X_tr = _drop_groups_columns(X_train_full, exp.include_groups)
        X_te = _drop_groups_columns(X_test_full, exp.include_groups)
        pipeline = build_sklearn_pipeline(tcfg.make_estimator())
        pipeline.fit(X_tr, y_train)
        mae = compute_metrics(y_test, pipeline.predict(X_te))["mae"]
        rows.append(
            {
                "experiment": exp.name,
                "include_groups": ",".join(exp.include_groups),
                "mae": round(float(mae), 2),
                "delta_mae_vs_full": round(float(mae - full_mae), 2),
            }
        )
    return pd.DataFrame(rows)


def run_model_selection(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config: EdaLabConfig,
    training_config: TrainingConfig | None = None,
) -> pd.DataFrame:
    """Quick holdout comparison across model_type candidates."""
    ms = config.model_selection
    if not ms.enabled:
        return pd.DataFrame()

    train_slice = _subsample(train_df, ms.max_train_rows)
    tcfg = training_config or load_training_config()

    baseline = BusinessBaseline().fit(train_slice)
    region_medians = {tuple(k.split("|")): v for k, v in baseline.lookup.items()}
    X_train = _feature_frame(train_slice, region_medians)
    X_test = _feature_frame(test_df, region_medians)
    y_train = train_slice[_TARGET].values.astype(float)
    y_test = test_df[_TARGET].values.astype(float)
    baseline_mae = compute_metrics(y_test, baseline.predict(test_df))["mae"]

    rows: list[dict[str, Any]] = [
        {
            "model": "business_baseline",
            "mae": round(float(baseline_mae), 2),
            "delta_mae_vs_best": None,
        }
    ]

    best_mae = baseline_mae
    for model_type in ms.candidates:
        cfg = tcfg.model_copy(update={"model_type": model_type})
        pipeline = build_sklearn_pipeline(cfg.make_estimator())
        pipeline.fit(X_train, y_train)
        mae = compute_metrics(y_test, pipeline.predict(X_test))["mae"]
        best_mae = min(best_mae, mae)
        rows.append({"model": model_type, "mae": round(float(mae), 2), "delta_mae_vs_best": None})

    out = pd.DataFrame(rows)
    out["delta_mae_vs_best"] = out["mae"].apply(lambda m: round(float(m - best_mae), 2))
    return out.sort_values("mae")


def residual_analysis_report(
    holdout: pd.DataFrame,
    config: EdaLabConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Segment residual summary and top worst predictions."""
    frame = holdout.copy()
    if "abs_error" not in frame.columns and "residual" in frame.columns:
        frame["abs_error"] = frame["residual"].abs()

    segment_frames: list[pd.DataFrame] = []
    for col in config.residual_analysis.segment_columns:
        if col not in frame.columns:
            continue
        seg = evaluate_by_segment(frame, "label_sale_price", "predicted_price", col)
        seg.insert(0, "segment_type", col)
        segment_frames.append(seg)

    segments = pd.concat(segment_frames, ignore_index=True) if segment_frames else pd.DataFrame()

    sort_cols = ["abs_error", "listing_id", "region", "property_type", "label_sale_price", "predicted_price"]
    present = [c for c in sort_cols if c in frame.columns]
    worst = frame.nlargest(config.residual_analysis.top_n_worst, "abs_error")[present].copy()
    return segments, worst


def run_shap_report(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config: EdaLabConfig,
    training_config: TrainingConfig | None = None,
) -> pd.DataFrame:
    """Train full model and return top SHAP features as a table."""
    if not config.shap.enabled:
        return pd.DataFrame()

    tcfg = training_config or load_training_config()
    train_slice = _subsample(train_df, config.feature_matrix.max_train_rows)

    baseline = BusinessBaseline().fit(train_slice)
    region_medians = {tuple(k.split("|")): v for k, v in baseline.lookup.items()}
    X_train = _feature_frame(train_slice, region_medians)
    X_test = _feature_frame(test_df, region_medians)
    y_train = train_slice[_TARGET].values.astype(float)

    pipeline = build_sklearn_pipeline(tcfg.make_estimator())
    pipeline.fit(X_train, y_train)

    sample_n = min(config.shap.max_samples, len(X_test))
    X_sample = X_test.sample(n=sample_n, random_state=42) if len(X_test) > sample_n else X_test
    summary = compute_shap_summary(
        pipeline,
        X_sample,
        max_features=config.shap.max_features,
    )
    if not summary.get("enabled"):
        return pd.DataFrame([{"error": summary.get("error", "SHAP disabled")}])

    return pd.DataFrame(summary["top_features"])


def run_eda_playbook(
    df: pd.DataFrame,
    *,
    dq_summary: dict[str, Any] | None = None,
    config: EdaLabConfig | None = None,
) -> dict[str, pd.DataFrame]:
    """Run all enabled EDA sections that only need the raw training frame."""
    cfg = config or load_eda_lab_config()
    results: dict[str, pd.DataFrame] = {}

    if cfg.sections.data_quality and dq_summary is not None:
        results["data_quality_gates"] = evaluate_data_quality_gates(dq_summary, cfg)
    if cfg.sections.univariate:
        results["univariate"] = univariate_profile(df, cfg)
    if cfg.sections.bivariate:
        results["bivariate"] = bivariate_vs_target(df, cfg)
    if cfg.sections.correlation:
        matrix, pairs = correlation_report(df, cfg)
        results["correlation_matrix"] = matrix
        results["correlation_flags"] = pairs
    if cfg.sections.business_hypotheses:
        results["business_hypotheses"] = evaluate_business_hypotheses(df, cfg)

    return results
