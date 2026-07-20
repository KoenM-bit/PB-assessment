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
import type { CategoryShare, CategoryTrendSeries } from "../types";

const LINE_COLORS = ["#2563eb", "#7c3aed", "#059669", "#d97706", "#dc2626", "#0891b2", "#4f46e5", "#be185d"];

type Props = {
  title: string;
  shares: CategoryShare[];
  trends: CategoryTrendSeries[];
  barNameKey: "region" | "type";
};

export function CategoryShareCharts({ title, shares, trends, barNameKey }: Props) {
  const barData = shares
    .filter((r) => r.count > 0)
    .map((r) => ({
      [barNameKey]: barNameKey === "type" ? r.label.replace(/_/g, " ") : r.label,
      live: Math.round(r.share * 1000) / 10,
      expected: r.expected_share != null ? Math.round(r.expected_share * 1000) / 10 : 0,
      trend_pp: r.trend_pp,
    }));

  const lineData = buildMultiLineChartData(trends);
  const activeSeries = trends.filter((s) => s.current_share_pct > 0).slice(0, 6);

  return (
    <div className="chart-panel category-share-panel">
      <div className="feature-skew-header">
        <h3>{title}</h3>
      </div>
      <p className="muted chart-caption">
        Bars = full window. Lines = daily share %. Trend badge = later days vs earlier days in window.
      </p>

      <div className="category-trend-chips">
        {shares
          .filter((s) => s.count > 0)
          .sort((a, b) => b.share - a.share)
          .slice(0, 8)
          .map((s) => (
            <span key={s.label} className="category-trend-chip">
              {barNameKey === "type" ? s.label.replace(/_/g, " ") : s.label}
              {": "}
              {(s.share * 100).toFixed(1)}%
              <TrendBadge trend_pp={s.trend_pp} />
            </span>
          ))}
      </div>

      <ResponsiveContainer width="100%" height={220}>
        <BarChart data={barData}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey={barNameKey} />
          <YAxis unit="%" />
          <Tooltip formatter={(v: number) => `${v}%`} />
          <Legend />
          <Bar dataKey="live" name="Live (window)" fill="#2563eb" />
          <Bar dataKey="expected" name="Uniform ref." fill="#94a3b8" />
        </BarChart>
      </ResponsiveContainer>

      {lineData.length > 1 && activeSeries.length > 0 && (
        <>
          <h4 className="category-trend-title">Daily share trend</h4>
          <ResponsiveContainer width="100%" height={200}>
            <LineChart data={lineData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="date" />
              <YAxis unit="%" domain={[0, "auto"]} />
              <Tooltip formatter={(v: number) => `${v}%`} />
              <Legend />
              {activeSeries.map((series, index) => (
                <Line
                  key={series.label}
                  type="monotone"
                  dataKey={series.label}
                  name={series.display_label}
                  stroke={LINE_COLORS[index % LINE_COLORS.length]}
                  strokeWidth={2}
                  dot={{ r: 3 }}
                />
              ))}
            </LineChart>
          </ResponsiveContainer>
        </>
      )}
    </div>
  );
}

function TrendBadge({ trend_pp }: { trend_pp: number | null }) {
  if (trend_pp == null) {
    return <span className="trend-badge flat"> →</span>;
  }
  if (Math.abs(trend_pp) < 0.5) {
    return <span className="trend-badge flat"> → {trend_pp.toFixed(1)}pp</span>;
  }
  if (trend_pp > 0) {
    return <span className="trend-badge up"> ↑ {trend_pp.toFixed(1)}pp</span>;
  }
  return <span className="trend-badge down"> ↓ {Math.abs(trend_pp).toFixed(1)}pp</span>;
}

function buildMultiLineChartData(series: CategoryTrendSeries[]) {
  const top = series.filter((s) => s.current_share_pct > 0).slice(0, 6);
  const dates = [...new Set(top.flatMap((s) => s.points.map((p) => p.date)))].sort();
  return dates.map((date) => {
    const row: Record<string, string | number> = { date: date.slice(5) };
    for (const s of top) {
      const pt = s.points.find((p) => p.date === date);
      row[s.label] = pt?.share_pct ?? 0;
    }
    return row;
  });
}
