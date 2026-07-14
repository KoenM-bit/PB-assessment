import { useEffect, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { api } from "../api/client";
import type { MetricSet, ModelComparison, MonitoringData } from "../types";
import { formatCurrency, formatDate, formatPercent } from "../utils/format";

export function MonitoringPage() {
  const [data, setData] = useState<MonitoringData | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    api.getMonitoring()
      .then(setData)
      .catch((err) => setError(err.message))
      .finally(() => setLoading(false));
  }, []);

  if (loading) return <div className="card">Loading monitoring data…</div>;
  if (error) return <div className="error">{error}</div>;
  if (!data) return null;

  const { summary, training, holdout_evaluation, live_evaluation, performance, data_quality, prediction_distribution, warnings } = data;

  const holdoutChartData = [
    {
      name: "Holdout test",
      model: Math.round(holdout_evaluation.model.mae),
      baseline: Math.round(holdout_evaluation.baseline.mae),
      sample_size: holdout_evaluation.model.sample_size,
    },
  ];

  const liveChartData = [
    {
      name: "Production",
      model: Math.round(live_evaluation.overall.model.mae),
      baseline: Math.round(live_evaluation.overall.baseline.mae),
      sample_size: live_evaluation.overall.model.sample_size,
    },
  ];

  const regionChartData = Object.entries(performance.by_region).map(([region, m]) => ({
    region,
    mae: Math.round(m.mae),
    sample_size: m.sample_size,
  }));

  const propertyChartData = Object.entries(performance.by_property_type).map(([type, m]) => ({
    type: type.replace(/_/g, " "),
    mae: Math.round(m.mae),
    sample_size: m.sample_size,
  }));

  return (
    <>
      {warnings.map((w) => (
        <div key={w} className="warning badge-warning">{w}</div>
      ))}

      <div className="card">
        <h2>Training Data Specs</h2>
        <p className="muted">
          Reference distribution from the last training run ({training.model_type},{" "}
          pipeline v{training.feature_pipeline_version}).
        </p>
        <div className="metrics-grid">
          <MetricCard label="Training rows" value={String(training.training_data_rows)} />
          <MetricCard label="Holdout test rows" value={String(training.test_rows)} />
          <MetricCard label="Regions" value={String(training.regions.length)} />
          <MetricCard label="Property types" value={String(training.property_types.length)} />
          <MetricCard
            label="Surface area"
            value={`${Math.round(training.surface_area_range.min)}–${Math.round(training.surface_area_range.max)} m²`}
            sub={`median ${Math.round(training.surface_area_range.median)} m²`}
          />
          <MetricCard
            label="Sale price"
            value={`${formatCurrency(training.price_range.min)} – ${formatCurrency(training.price_range.max)}`}
            sub={`median ${formatCurrency(training.price_range.median)}`}
          />
          <MetricCard label="Validation" value={training.validation_approach} />
          <MetricCard
            label="Trained"
            value={formatDate(training.training_date)}
            sub={`commit ${training.git_commit.slice(0, 8)}`}
          />
        </div>

        {training.feature_bounds_summary.length > 0 && (
          <div className="table-wrap">
            <h3>Feature ranges (p01 – p99)</h3>
            <table>
              <thead>
                <tr>
                  <th>Feature</th>
                  <th>Lower (p01)</th>
                  <th>Upper (p99)</th>
                </tr>
              </thead>
              <tbody>
                {training.feature_bounds_summary.map((row) => (
                  <tr key={row.feature}>
                    <td>{row.feature.replace(/_/g, " ")}</td>
                    <td>{formatFeatureValue(row.feature, row.p01)}</td>
                    <td>{formatFeatureValue(row.feature, row.p99)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      <div className="card">
        <h2>Model vs Business Baseline</h2>
        <p className="muted">
          Business baseline = median price/m² by region × property type × surface area.
        </p>

        <div className="comparison-grid">
          <ComparisonPanel
            title="Holdout test set"
            comparison={holdout_evaluation}
            subtitle={`n=${holdout_evaluation.model.sample_size} (offline evaluation at training time)`}
          />
          <ComparisonPanel
            title="Live production"
            comparison={live_evaluation.overall}
            subtitle={
              live_evaluation.overall.model.sample_size > 0
                ? `n=${live_evaluation.overall.model.sample_size} labelled predictions with actual sales`
                : "Record actual sales on the Predictions & Sales page to populate live metrics"
            }
          />
        </div>

        <div className="chart-row">
          <div className="chart-panel">
            <h3>MAE — holdout test</h3>
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={holdoutChartData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" />
                <YAxis />
                <Tooltip formatter={(v: number) => formatCurrency(v)} />
                <Legend />
                <Bar dataKey="model" name="ML model" fill="#2563eb" />
                <Bar dataKey="baseline" name="Business baseline" fill="#94a3b8" />
              </BarChart>
            </ResponsiveContainer>
          </div>
          {live_evaluation.overall.model.sample_size > 0 && (
            <div className="chart-panel">
              <h3>MAE — live production</h3>
              <ResponsiveContainer width="100%" height={220}>
                <BarChart data={liveChartData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="name" />
                  <YAxis />
                  <Tooltip formatter={(v: number) => formatCurrency(v)} />
                  <Legend />
                  <Bar dataKey="model" name="ML model" fill="#7c3aed" />
                  <Bar dataKey="baseline" name="Business baseline" fill="#94a3b8" />
                </BarChart>
              </ResponsiveContainer>
            </div>
          )}
        </div>
      </div>

      <div className="card">
        <h2>Production Monitoring</h2>
        <p>
          Active model: <span className="badge">{summary.active_model_version}</span>
        </p>
        <div className="metrics-grid">
          <MetricCard label="Total predictions" value={String(summary.total_predictions)} />
          <MetricCard label="With actual sales" value={String(summary.labelled_predictions)} />
          <MetricCard
            label="Live MAE"
            value={formatCurrency(performance.overall.mae)}
            sub={performance.overall.is_reliable ? `n=${performance.overall.sample_size}` : `n=${performance.overall.sample_size} (low)`}
          />
          <MetricCard label="Live RMSE" value={formatCurrency(performance.overall.rmse)} />
          <MetricCard label="Live bias" value={formatCurrency(performance.overall.bias)} />
          <MetricCard label="Avg prediction" value={formatCurrency(prediction_distribution.mean)} />
          <MetricCard label="Out-of-range rate" value={`${(data_quality.out_of_range_rate * 100).toFixed(1)}%`} />
        </div>
      </div>

      {regionChartData.length > 0 && (
        <div className="card">
          <h3>Live MAE by Region</h3>
          <ResponsiveContainer width="100%" height={250}>
            <BarChart data={regionChartData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="region" />
              <YAxis />
              <Tooltip formatter={(v: number) => formatCurrency(v)} />
              <Bar dataKey="mae" fill="#2563eb" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}

      {propertyChartData.length > 0 && (
        <div className="card">
          <h3>Live MAE by Property Type</h3>
          <ResponsiveContainer width="100%" height={250}>
            <BarChart data={propertyChartData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="type" />
              <YAxis />
              <Tooltip formatter={(v: number) => formatCurrency(v)} />
              <Bar dataKey="mae" fill="#7c3aed" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}
    </>
  );
}

function MetricCard({ label, value, sub }: { label: string; value: string; sub?: string }) {
  return (
    <div className="metric-card">
      <div className="metric-value">{value}</div>
      <div className="metric-label">{label}</div>
      {sub && <div className="metric-label">{sub}</div>}
    </div>
  );
}

function ComparisonPanel({
  title,
  comparison,
  subtitle,
}: {
  title: string;
  comparison: ModelComparison;
  subtitle: string;
}) {
  return (
    <div className="comparison-panel">
      <h3>{title}</h3>
      <p className="muted">{subtitle}</p>
      <div className={`comparison-verdict ${comparison.beats_baseline ? "positive" : "negative"}`}>
        {comparison.model.sample_size === 0
          ? "No labelled data yet"
          : comparison.beats_baseline
            ? `Model beats baseline by ${comparison.mae_improvement_pct.toFixed(1)}% MAE`
            : `Baseline is better by ${Math.abs(comparison.mae_improvement_pct).toFixed(1)}% MAE`}
      </div>
      <table>
        <thead>
          <tr>
            <th>Metric</th>
            <th>ML model</th>
            <th>Baseline</th>
          </tr>
        </thead>
        <tbody>
          <ComparisonRow label="MAE" model={comparison.model} baseline={comparison.baseline} field="mae" currency />
          <ComparisonRow label="RMSE" model={comparison.model} baseline={comparison.baseline} field="rmse" currency />
          <ComparisonRow label="Bias" model={comparison.model} baseline={comparison.baseline} field="bias" currency />
          <ComparisonRow label="MAPE" model={comparison.model} baseline={comparison.baseline} field="mape" percent />
        </tbody>
      </table>
    </div>
  );
}

function ComparisonRow({
  label,
  model,
  baseline,
  field,
  currency,
  percent,
}: {
  label: string;
  model: MetricSet;
  baseline: MetricSet;
  field: keyof MetricSet;
  currency?: boolean;
  percent?: boolean;
}) {
  const format = (value: number) => {
    if (currency) return formatCurrency(value);
    if (percent) return formatPercent(value);
    return String(value);
  };

  return (
    <tr>
      <td>{label}</td>
      <td>{format(model[field] as number)}</td>
      <td>{format(baseline[field] as number)}</td>
    </tr>
  );
}

function formatFeatureValue(feature: string, value: number): string {
  if (feature.includes("price") || feature.includes("surface_x")) {
    return formatCurrency(value);
  }
  if (feature === "dist_to_city_centre_km") {
    return `${value.toFixed(1)} km`;
  }
  if (feature === "surface_area") {
    return `${value.toFixed(0)} m²`;
  }
  return value.toFixed(1);
}
