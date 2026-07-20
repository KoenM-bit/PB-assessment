import {
  CartesianGrid,
  Cell,
  ComposedChart,
  Line,
  LineChart,
  ReferenceArea,
  ResponsiveContainer,
  Scatter,
  Tooltip,
  XAxis,
  YAxis,
  ZAxis,
} from "recharts";
import type { FeatureDistributionViz, RequestMonitoring } from "../types";
import { formatFeatureValue } from "../utils/featureFormat";

type Props = {
  requests: RequestMonitoring;
};

export function FeatureSkewCharts({ requests }: Props) {
  if (!requests.feature_distributions?.length) return null;

  const trendSeries = buildTrendSeries(requests.feature_distributions);

  return (
    <div className="feature-skew-section">
      <h3>Numeric inputs — range check &amp; drift</h3>
      <p className="muted">
        Green band = training p01–p99. Each dot is one logged request (feature value only, no addresses).
        Normalized view maps that band to 0–100% so you can spot outliers without reading units.
      </p>

      <div className="feature-skew-grid">
        {requests.feature_distributions.map((dist) => (
          <FeatureRangePanel key={dist.feature} dist={dist} />
        ))}
      </div>

      {trendSeries.length > 0 && (
        <div className="chart-panel feature-trend-panel">
          <h4>% outside training range (daily trend)</h4>
          <ResponsiveContainer width="100%" height={220}>
            <LineChart data={trendSeries}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="date" />
              <YAxis unit="%" domain={[0, "auto"]} />
              <Tooltip formatter={(v: number) => `${v.toFixed(1)}%`} />
              {requests.feature_distributions.map((dist, i) => (
                <Line
                  key={dist.feature}
                  type="monotone"
                  dataKey={dist.feature}
                  name={dist.label}
                  stroke={TREND_COLORS[i % TREND_COLORS.length]}
                  strokeWidth={2}
                  dot={false}
                />
              ))}
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}
    </div>
  );
}

const TREND_COLORS = ["#2563eb", "#7c3aed", "#059669", "#d97706", "#dc2626"];

function FeatureRangePanel({ dist }: { dist: FeatureDistributionViz }) {
  const scatterData = dist.points.map((p) => ({
    value: p.value,
    jitter: p.jitter,
    in_range: p.in_range,
    fill: p.in_range ? "#2563eb" : "#dc2626",
  }));

  const normData = dist.points.map((p) => ({
    position_pct: p.position_pct,
    jitter: p.jitter,
    in_range: p.in_range,
    fill: p.in_range ? "#059669" : "#dc2626",
  }));

  const xPad = (dist.training_p99 - dist.training_p01) * 0.08 || 1;
  const xMin = dist.training_p01 - xPad;
  const xMax = dist.training_p99 + xPad;

  return (
    <div className="feature-skew-panel">
      <div className="feature-skew-header">
        <strong>{dist.label}</strong>
        <span className={`badge ${dist.pct_out_of_range > 5 ? "badge-warning" : ""}`}>
          {dist.pct_out_of_range.toFixed(1)}% outside band
        </span>
      </div>
      <p className="muted feature-skew-sub">
        Training p01–p99: {formatFeatureValue(dist.feature, dist.training_p01)} –{" "}
        {formatFeatureValue(dist.feature, dist.training_p99)} · recent mean{" "}
        {formatFeatureValue(dist.feature, dist.recent_mean)} (n={dist.sample_size})
      </p>

      <div className="feature-dual-charts">
        <div>
          <div className="feature-chart-label">Actual scale</div>
          <ResponsiveContainer width="100%" height={130}>
            <ComposedChart data={scatterData} margin={{ top: 8, right: 8, left: 0, bottom: 0 }}>
              <XAxis
                type="number"
                dataKey="value"
                domain={[xMin, xMax]}
                tickFormatter={(v) => formatFeatureValue(dist.feature, v)}
                fontSize={11}
              />
              <YAxis type="number" dataKey="jitter" hide domain={[0, 1]} />
              <ZAxis range={[40, 40]} />
              <ReferenceArea
                x1={dist.training_p01}
                x2={dist.training_p99}
                fill="#dcfce7"
                fillOpacity={0.9}
                stroke="#86efac"
              />
              <Tooltip
                formatter={(v: number, _n, item) => {
                  const payload = item?.payload as { in_range?: boolean };
                  return [
                    formatFeatureValue(dist.feature, v),
                    payload?.in_range ? "in range" : "outside",
                  ];
                }}
              />
              <Scatter data={scatterData} shape="circle">
                {scatterData.map((entry, index) => (
                  <Cell key={index} fill={entry.fill} />
                ))}
              </Scatter>
            </ComposedChart>
          </ResponsiveContainer>
        </div>
        <div>
          <div className="feature-chart-label">Normalized (0% = p01, 100% = p99)</div>
          <ResponsiveContainer width="100%" height={130}>
            <ComposedChart data={normData} margin={{ top: 8, right: 8, left: 0, bottom: 0 }}>
              <XAxis type="number" dataKey="position_pct" domain={[-15, 115]} unit="%" fontSize={11} />
              <YAxis type="number" dataKey="jitter" hide domain={[0, 1]} />
              <ZAxis range={[40, 40]} />
              <ReferenceArea x1={0} x2={100} fill="#dcfce7" fillOpacity={0.9} stroke="#86efac" />
              <Tooltip formatter={(v: number) => `${v.toFixed(0)}%`} />
              <Scatter data={normData} shape="circle">
                {normData.map((entry, index) => (
                  <Cell key={index} fill={entry.fill} />
                ))}
              </Scatter>
            </ComposedChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
}

function buildTrendSeries(distributions: FeatureDistributionViz[]) {
  const dates = new Set<string>();
  for (const dist of distributions) {
    for (const row of dist.daily_trend) dates.add(row.date);
  }
  return [...dates]
    .sort()
    .map((date) => {
      const row: Record<string, string | number> = { date: date.slice(5) };
      for (const dist of distributions) {
        const day = dist.daily_trend.find((d) => d.date === date);
        row[dist.feature] = day?.pct_outside ?? 0;
      }
      return row;
    });
}
