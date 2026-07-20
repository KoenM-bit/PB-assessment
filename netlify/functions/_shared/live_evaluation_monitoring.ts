export interface MaeTrendPoint {
  date: string;
  mae: number;
  count: number;
}

export interface MaeTrendSeries {
  label: string;
  display_label: string;
  points: MaeTrendPoint[];
  /** Second-half avg MAE minus first-half (negative = improving). */
  trend_eur: number | null;
  current_mae: number;
  sample_size: number;
}

export interface MaeSegmentBar {
  label: string;
  display_label: string;
  mae: number;
  sample_size: number;
  trend_eur: number | null;
}

export interface LiveLabelledSale {
  prediction_id: string;
  address: string;
  region: string;
  property_type: string;
  surface_area: number;
  predicted_price: number;
  baseline_price: number;
  actual_sale_price: number;
  model_abs_error: number;
  baseline_abs_error: number;
  model_pct_error: number;
  prediction_date: string;
  sale_date: string | null;
  beats_baseline: boolean;
}

export interface LiveLabelledMonitoring {
  sample_size: number;
  by_region: MaeSegmentBar[];
  by_property_type: MaeSegmentBar[];
  region_mae_trends: MaeTrendSeries[];
  property_type_mae_trends: MaeTrendSeries[];
  items: LiveLabelledSale[];
}

export interface LabelledPredictionRow {
  prediction_id: string;
  address: string;
  region: string;
  property_type: string;
  surface_area: number;
  predicted: number;
  actual: number;
  baseline: number;
  prediction_timestamp: string;
  sale_date: string | null;
}

function roundMoney(value: number): number {
  return Math.round(value);
}

function computeTrendEur(points: MaeTrendPoint[]): number | null {
  const withData = points.filter((p) => p.count > 0);
  if (withData.length < 2) return null;
  const mid = Math.floor(withData.length / 2);
  const first = withData.slice(0, mid);
  const second = withData.slice(mid);
  const avg = (pts: MaeTrendPoint[]) =>
    pts.length > 0 ? pts.reduce((sum, p) => sum + p.mae, 0) / pts.length : 0;
  return roundMoney(avg(second) - avg(first));
}

function buildMaeTrendSeries(
  rows: LabelledPredictionRow[],
  getLabel: (r: LabelledPredictionRow) => string,
): MaeTrendSeries[] {
  const byDay = new Map<string, LabelledPredictionRow[]>();
  for (const row of rows) {
    const day = row.prediction_timestamp.slice(0, 10);
    const bucket = byDay.get(day) ?? [];
    bucket.push(row);
    byDay.set(day, bucket);
  }
  const days = [...byDay.keys()].sort();
  const labels = [...new Set(rows.map(getLabel))].sort();

  return labels
    .map((label) => {
      const segmentRows = rows.filter((r) => getLabel(r) === label);
      const points: MaeTrendPoint[] = days.map((day) => {
        const dayRows = (byDay.get(day) ?? []).filter((r) => getLabel(r) === label);
        if (dayRows.length === 0) {
          return { date: day, mae: 0, count: 0 };
        }
        const mae =
          dayRows.reduce((sum, r) => sum + Math.abs(r.actual - r.predicted), 0) /
          dayRows.length;
        return { date: day, mae: roundMoney(mae), count: dayRows.length };
      });
      const overallMae =
        segmentRows.reduce((sum, r) => sum + Math.abs(r.actual - r.predicted), 0) /
        segmentRows.length;
      return {
        label,
        display_label: label.replace(/_/g, " "),
        points,
        trend_eur: computeTrendEur(points),
        current_mae: roundMoney(overallMae),
        sample_size: segmentRows.length,
      };
    })
    .filter((s) => s.sample_size > 0)
    .sort((a, b) => b.sample_size - a.sample_size);
}

function segmentBars(
  rows: LabelledPredictionRow[],
  getLabel: (r: LabelledPredictionRow) => string,
  trends: MaeTrendSeries[],
): MaeSegmentBar[] {
  const trendMap = new Map(trends.map((t) => [t.label, t.trend_eur]));
  const labels = [...new Set(rows.map(getLabel))].sort();
  return labels.map((label) => {
    const segment = rows.filter((r) => getLabel(r) === label);
    const mae =
      segment.reduce((sum, r) => sum + Math.abs(r.actual - r.predicted), 0) / segment.length;
    return {
      label,
      display_label: label.replace(/_/g, " "),
      mae: roundMoney(mae),
      sample_size: segment.length,
      trend_eur: trendMap.get(label) ?? null,
    };
  });
}

export function buildLiveLabelledMonitoring(
  rows: LabelledPredictionRow[],
): LiveLabelledMonitoring {
  const regionTrends = buildMaeTrendSeries(rows, (r) => r.region);
  const propertyTrends = buildMaeTrendSeries(rows, (r) => r.property_type);

  const items: LiveLabelledSale[] = rows
    .map((r) => {
      const modelAbs = Math.abs(r.actual - r.predicted);
      const baselineAbs = Math.abs(r.actual - r.baseline);
      return {
        prediction_id: r.prediction_id,
        address: r.address,
        region: r.region,
        property_type: r.property_type,
        surface_area: r.surface_area,
        predicted_price: r.predicted,
        baseline_price: roundMoney(r.baseline),
        actual_sale_price: r.actual,
        model_abs_error: roundMoney(modelAbs),
        baseline_abs_error: roundMoney(baselineAbs),
        model_pct_error: Math.round((modelAbs / r.actual) * 1000) / 10,
        prediction_date: r.prediction_timestamp,
        sale_date: r.sale_date,
        beats_baseline: modelAbs < baselineAbs,
      };
    })
    .sort((a, b) => b.prediction_date.localeCompare(a.prediction_date));

  return {
    sample_size: rows.length,
    by_region: segmentBars(rows, (r) => r.region, regionTrends),
    by_property_type: segmentBars(rows, (r) => r.property_type, propertyTrends),
    region_mae_trends: regionTrends,
    property_type_mae_trends: propertyTrends,
    items,
  };
}
