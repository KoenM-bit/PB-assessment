import { useEffect, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { FeatureSkewCharts } from "../components/FeatureSkewCharts";
import { api } from "../api/client";
import type { MetricSet, ModelComparison, MonitoringData } from "../types";
import { formatCurrency, formatDate, formatDurationMs, formatPercent } from "../utils/format";
import { formatFeatureValue } from "../utils/featureFormat";

const EMPTY_LATENCY = { sample_size: 0, avg_ms: 0, p50_ms: 0, p95_ms: 0, max_ms: 0 };

function normalizeInfrastructure(
  raw: MonitoringData["infrastructure"] | undefined,
  servingEndpointFallback: string,
): MonitoringData["infrastructure"] {
  return {
    request_count: raw?.request_count ?? 0,
    error_rate: raw?.error_rate ?? 0,
    timeout_rate: raw?.timeout_rate ?? 0,
    api_latency: raw?.api_latency ?? EMPTY_LATENCY,
    fallback_rate: raw?.fallback_rate ?? 0,
    peer_fallback_rate: raw?.peer_fallback_rate ?? 0,
    daily: raw?.daily ?? [],
    history: raw?.history ?? [],
    recent_requests: raw?.recent_requests ?? [],
    databricks_endpoint: raw?.databricks_endpoint ?? null,
    serving_endpoint: raw?.serving_endpoint ?? servingEndpointFallback,
  };
}

const EMPTY_REQUEST_MONITORING: MonitoringData["request_monitoring"] = {
  sample_size: 0,
  window_label: "recent predictions",
  by_region: [],
  by_property_type: [],
  numeric_features: [],
  feature_distributions: [],
  warnings: [],
};

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

  const { summary, training, holdout_evaluation, live_evaluation, performance, data_quality, prediction_distribution, warnings, feature_monitoring, request_monitoring } = data;
  const infrastructure = normalizeInfrastructure(data.infrastructure, "house-price-serving");
  const requests = request_monitoring ?? EMPTY_REQUEST_MONITORING;
  const deployStaleApi =
    data.infrastructure != null && !("api_latency" in data.infrastructure);

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

  const latencyHistory = infrastructure.history.map((row) => ({
    ...row,
    label: row.date.slice(5),
  }));

  const dailyVolume = infrastructure.daily.map((row) => ({
    date: row.date.slice(5),
    requests: row.request_count,
    fallbacks: row.fallback_count,
  }));

  const endpoint = infrastructure.databricks_endpoint;

  return (
    <>
      {warnings.map((w) => (
        <div key={w} className="warning badge-warning">{w}</div>
      ))}

      {deployStaleApi && (
        <div className="warning badge-warning">
          API is not on the latest monitoring version yet — redeploy Netlify functions (merge to staging or trigger deploy). Performance cards need the updated /api/monitoring.
        </div>
      )}

      <div className="card">
        <h2>Databricks &amp; API Performance</h2>
        <p className="muted">
          Live metrics from endpoint{" "}
          <span className="badge">{infrastructure.serving_endpoint}</span>
          {endpoint?.has_metrics
            ? " (Databricks metrics API)"
            : endpoint?.available
              ? " — Databricks metrics API reachable but no telemetry yet (endpoint idle or metrics reset). API latency below is from gold.predictions."
              : " — endpoint metrics unavailable; showing stored prediction latencies from gold.predictions"}
        </p>

        <div className="metrics-grid">
          {endpoint?.has_metrics && (
            <>
              <MetricCard
                label="Endpoint requests (total)"
                value={String(endpoint.request_count_total)}
              />
              <MetricCard
                label="Serving latency p50"
                value={formatDurationMs(endpoint.latency_p50_ms)}
                sub={
                  endpoint.latency_p50_ms != null
                    ? "inside Databricks"
                    : "not in metrics export"
                }
              />
              <MetricCard
                label="Serving latency p99"
                value={formatDurationMs(endpoint.latency_p99_ms)}
                sub={
                  endpoint.latency_p99_ms != null
                    ? "inside Databricks"
                    : "not in metrics export"
                }
              />
              <MetricCard
                label="CPU usage"
                value={endpoint.cpu_usage_pct != null ? `${endpoint.cpu_usage_pct}%` : "—"}
              />
              <MetricCard
                label="Memory usage"
                value={endpoint.memory_usage_pct != null ? `${endpoint.memory_usage_pct}%` : "—"}
              />
              <MetricCard
                label="4xx / 5xx errors"
                value={`${endpoint.error_4xx_total} / ${endpoint.error_5xx_total}`}
                sub={`error rate ${formatPercent(infrastructure.error_rate * 100)}`}
              />
            </>
          )}
          <MetricCard
            label="API latency p50"
            value={formatDurationMs(infrastructure.api_latency.p50_ms)}
            sub={`n=${infrastructure.api_latency.sample_size} recent predictions`}
          />
          <MetricCard
            label="API latency p95"
            value={formatDurationMs(infrastructure.api_latency.p95_ms)}
            sub={`avg ${formatDurationMs(infrastructure.api_latency.avg_ms)}`}
          />
          <MetricCard
            label="Max API latency"
            value={formatDurationMs(infrastructure.api_latency.max_ms)}
          />
          <MetricCard
            label="Baseline fallback rate"
            value={formatPercent(infrastructure.fallback_rate * 100)}
            sub={
              infrastructure.peer_fallback_rate > 0
                ? `peer serving ${formatPercent(infrastructure.peer_fallback_rate * 100)} (still ML)`
                : "business baseline only"
            }
          />
          <MetricCard label="Tracked requests" value={String(infrastructure.request_count)} />
        </div>

        {infrastructure.api_latency.sample_size === 0 && (
          <p className="muted">
            No latency data yet — successful predictions are required in <code>gold.predictions</code>.
          </p>
        )}

        {infrastructure.recent_requests.length > 0 && (
          <div className="table-wrap">
            <h3>Last {infrastructure.recent_requests.length} API requests (end-to-end latency)</h3>
            <p className="muted">
              Per prediction stored in Databricks — same source as the trend chart below (not Databricks endpoint p50/p99).
            </p>
            <table>
              <thead>
                <tr>
                  <th>Time</th>
                  <th>Latency</th>
                  <th>Model</th>
                  <th>Route</th>
                  <th>Prediction ID</th>
                </tr>
              </thead>
              <tbody>
                {infrastructure.recent_requests.map((row) => (
                  <tr key={row.prediction_id}>
                    <td>{formatDate(row.timestamp)}</td>
                    <td>{formatDurationMs(row.latency_ms)}</td>
                    <td>{row.model_version}</td>
                    <td>
                      <span className={`badge ${row.serving_route === "primary" ? "" : "badge-warning"}`}>
                        {row.serving_route}
                      </span>
                    </td>
                    <td className="muted">{row.prediction_id.slice(0, 8)}…</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}

        {latencyHistory.length > 0 && (
          <div className="chart-row">
            <div className="chart-panel">
              <h3>Daily API latency (p50 / p95)</h3>
              <p className="muted chart-caption">
                Aggregated from <code>gold.predictions.serving_latency_ms</code> per day.
              </p>
              <ResponsiveContainer width="100%" height={240}>
                <LineChart data={latencyHistory}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="label" />
                  <YAxis tickFormatter={(v) => `${v} ms`} />
                  <Tooltip formatter={(v: number) => formatDurationMs(v)} />
                  <Legend />
                  <Line type="monotone" dataKey="p50_ms" name="p50" stroke="#2563eb" strokeWidth={2} dot={false} />
                  <Line type="monotone" dataKey="p95_ms" name="p95" stroke="#7c3aed" strokeWidth={2} dot={false} />
                </LineChart>
              </ResponsiveContainer>
            </div>
            {dailyVolume.length > 0 && (
              <div className="chart-panel">
                <h3>Request volume</h3>
                <ResponsiveContainer width="100%" height={240}>
                  <BarChart data={dailyVolume}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="date" />
                    <YAxis allowDecimals={false} />
                    <Tooltip />
                    <Legend />
                    <Bar dataKey="requests" name="Predictions" fill="#2563eb" />
                    <Bar dataKey="fallbacks" name="Baseline fallbacks" fill="#f59e0b" />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            )}
          </div>
        )}
      </div>

      <div className="card">
        <h2>Incoming requests vs training</h2>
        <p className="muted">
          Distribution of recent API payloads ({requests.window_label}, n={requests.sample_size}) compared to
          training coverage. Large skew or out-of-range rates can signal data drift before model quality drops.
        </p>

        {requests.sample_size === 0 ? (
          <p className="muted">No logged requests yet.</p>
        ) : (
          <>
            <div className="chart-row">
              <div className="chart-panel">
                <h3>Share by region (%)</h3>
                <ResponsiveContainer width="100%" height={260}>
                  <BarChart
                    data={requests.by_region
                      .filter((r) => r.count > 0)
                      .map((r) => ({
                        region: r.label,
                        live: Math.round(r.share * 1000) / 10,
                        expected: r.expected_share != null ? Math.round(r.expected_share * 1000) / 10 : 0,
                      }))}
                  >
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="region" />
                    <YAxis unit="%" />
                    <Tooltip formatter={(v: number) => `${v}%`} />
                    <Legend />
                    <Bar dataKey="live" name="Live requests" fill="#2563eb" />
                    <Bar dataKey="expected" name="Uniform training ref." fill="#94a3b8" />
                  </BarChart>
                </ResponsiveContainer>
              </div>
              <div className="chart-panel">
                <h3>Share by property type (%)</h3>
                <ResponsiveContainer width="100%" height={260}>
                  <BarChart
                    data={requests.by_property_type
                      .filter((r) => r.count > 0)
                      .map((r) => ({
                        type: r.label.replace(/_/g, " "),
                        live: Math.round(r.share * 1000) / 10,
                        expected: r.expected_share != null ? Math.round(r.expected_share * 1000) / 10 : 0,
                      }))}
                  >
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="type" />
                    <YAxis unit="%" />
                    <Tooltip formatter={(v: number) => `${v}%`} />
                    <Legend />
                    <Bar dataKey="live" name="Live requests" fill="#7c3aed" />
                    <Bar dataKey="expected" name="Uniform training ref." fill="#94a3b8" />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>

            <FeatureSkewCharts requests={requests} />

            {feature_monitoring.length > 0 && (
              <div className="table-wrap">
                <h3>Databricks feature monitoring (latest job run)</h3>
                <table>
                  <thead>
                    <tr>
                      <th>Feature</th>
                      <th>Recent mean</th>
                      <th>Reference mean</th>
                      <th>% out of range</th>
                      <th>Drift score</th>
                    </tr>
                  </thead>
                  <tbody>
                    {feature_monitoring.map((row) => (
                      <tr key={row.feature_name}>
                        <td>{row.feature_name.replace(/_/g, " ")}</td>
                        <td>{row.recent_mean.toFixed(2)}</td>
                        <td>{row.reference_mean.toFixed(2)}</td>
                        <td>{row.pct_out_of_range.toFixed(1)}%</td>
                        <td>{row.drift_score.toFixed(1)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </>
        )}
      </div>

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
