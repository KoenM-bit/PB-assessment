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
import type { MaeSegmentBar, MaeTrendSeries } from "../types";
import { formatCurrency } from "../utils/format";

const LINE_COLORS = ["#2563eb", "#7c3aed", "#059669", "#d97706", "#dc2626", "#0891b2", "#4f46e5", "#be185d"];

type Props = {
  title: string;
  segments: MaeSegmentBar[];
  trends: MaeTrendSeries[];
  nameKey: "region" | "type";
};

export function LiveMaeCharts({ title, segments, trends, nameKey }: Props) {
  const barData = segments
    .filter((s) => s.sample_size > 0)
    .map((s) => ({
      [nameKey]: s.display_label,
      mae: s.mae,
      sample_size: s.sample_size,
      trend_eur: s.trend_eur,
    }));

  const lineData = buildMaeLineData(trends);
  const activeSeries = trends.filter((s) => s.sample_size > 0).slice(0, 6);

  return (
    <div className="chart-panel category-share-panel">
      <h3>{title}</h3>
      <p className="muted chart-caption">
        Bars = MAE over all labelled sales in window. Lines = daily MAE per segment (lower is better).
      </p>

      <div className="category-trend-chips">
        {segments
          .filter((s) => s.sample_size > 0)
          .sort((a, b) => b.sample_size - a.sample_size)
          .slice(0, 8)
          .map((s) => (
            <span key={s.label} className="category-trend-chip">
              {s.display_label}
              {": "}
              {formatCurrency(s.mae)}
              <MaeTrendBadge trend_eur={s.trend_eur} />
            </span>
          ))}
      </div>

      <ResponsiveContainer width="100%" height={220}>
        <BarChart data={barData}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey={nameKey} />
          <YAxis tickFormatter={(v) => `€${Math.round(v / 1000)}k`} />
          <Tooltip formatter={(v: number) => formatCurrency(v)} />
          <Legend />
          <Bar dataKey="mae" name="Live MAE" fill="#2563eb" />
        </BarChart>
      </ResponsiveContainer>

      {lineData.length > 1 && activeSeries.length > 0 && (
        <>
          <h4 className="category-trend-title">Daily MAE trend</h4>
          <ResponsiveContainer width="100%" height={200}>
            <LineChart data={lineData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="date" />
              <YAxis tickFormatter={(v) => `€${Math.round(v / 1000)}k`} />
              <Tooltip formatter={(v: number) => formatCurrency(v)} />
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
                  connectNulls
                />
              ))}
            </LineChart>
          </ResponsiveContainer>
        </>
      )}
    </div>
  );
}

export function MaeTrendBadge({ trend_eur }: { trend_eur: number | null }) {
  if (trend_eur == null) {
    return <span className="trend-badge flat"> →</span>;
  }
  if (Math.abs(trend_eur) < 750) {
    return <span className="trend-badge flat"> → {formatCurrency(Math.abs(trend_eur))}</span>;
  }
  if (trend_eur < 0) {
    return (
      <span className="trend-badge up" title="MAE improving">
        {" "}
        ↓ {formatCurrency(Math.abs(trend_eur))}
      </span>
    );
  }
  return (
    <span className="trend-badge down" title="MAE worsening">
      {" "}
      ↑ {formatCurrency(trend_eur)}
    </span>
  );
}

function buildMaeLineData(series: MaeTrendSeries[]) {
  const top = series.filter((s) => s.sample_size > 0).slice(0, 6);
  const dates = [...new Set(top.flatMap((s) => s.points.map((p) => p.date)))].sort();
  return dates.map((date) => {
    const row: Record<string, string | number | null> = { date: date.slice(5) };
    for (const s of top) {
      const pt = s.points.find((p) => p.date === date);
      row[s.label] = pt && pt.count > 0 ? pt.mae : null;
    }
    return row;
  });
}
