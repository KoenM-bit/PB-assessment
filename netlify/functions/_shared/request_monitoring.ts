import type { StoredPrediction } from "./databricks.js";
import type { TrainingManifest } from "./training_manifest.js";

export interface CategoryShare {
  label: string;
  count: number;
  share: number;
  expected_share: number | null;
  skew_pp: number | null;
  /** Recent vs earlier daily share in window (percentage points). */
  trend_pp: number | null;
}

export interface CategoryTrendPoint {
  date: string;
  share_pct: number;
  count: number;
}

export interface CategoryTrendSeries {
  label: string;
  display_label: string;
  points: CategoryTrendPoint[];
  trend_pp: number | null;
  current_share_pct: number;
}

export interface FeatureDriftRow {
  feature: string;
  recent_mean: number;
  reference_mean: number;
  pct_out_of_range: number;
  sample_size: number;
}

export interface FeaturePointViz {
  /** Anonymous sequence 1..n (not prediction id). */
  index: number;
  value: number;
  in_range: boolean;
  /** 0 = training p01, 100 = training p99 (can be &lt;0 or &gt;100). */
  position_pct: number;
  jitter: number;
  day: string;
}

export interface FeatureDailyTrend {
  date: string;
  mean: number;
  pct_outside: number;
  n: number;
}

export interface FeatureDistributionViz {
  feature: string;
  label: string;
  training_p01: number;
  training_p99: number;
  recent_mean: number;
  pct_out_of_range: number;
  sample_size: number;
  points: FeaturePointViz[];
  daily_trend: FeatureDailyTrend[];
}

export interface RequestMonitoring {
  sample_size: number;
  window_label: string;
  by_region: CategoryShare[];
  by_property_type: CategoryShare[];
  region_trends: CategoryTrendSeries[];
  property_type_trends: CategoryTrendSeries[];
  numeric_features: FeatureDriftRow[];
  feature_distributions: FeatureDistributionViz[];
  warnings: string[];
}

function parsePayload(raw: string): Record<string, unknown> {
  try {
    return JSON.parse(raw) as Record<string, unknown>;
  } catch {
    return {};
  }
}

function categoryShares(
  labels: string[],
  expectedLabels: string[],
): CategoryShare[] {
  const total = labels.length;
  const counts = new Map<string, number>();
  for (const label of labels) {
    counts.set(label, (counts.get(label) ?? 0) + 1);
  }

  const expectedShare =
    expectedLabels.length > 0 ? 1 / expectedLabels.length : null;

  const allLabels = [...new Set([...expectedLabels, ...counts.keys()])].sort();

  return allLabels.map((label) => {
    const count = counts.get(label) ?? 0;
    const share = total > 0 ? count / total : 0;
    const skew_pp =
      expectedShare != null ? (share - expectedShare) * 100 : null;
    return {
      label,
      count,
      share,
      expected_share: expectedShare,
      skew_pp: skew_pp != null ? Math.round(skew_pp * 10) / 10 : null,
      trend_pp: null,
    };
  });
}

function computeTrendPp(points: CategoryTrendPoint[]): number | null {
  if (points.length < 2) return null;
  const mid = Math.floor(points.length / 2);
  const first = points.slice(0, mid);
  const second = points.slice(mid);
  const avg = (pts: CategoryTrendPoint[]) =>
    pts.length > 0 ? pts.reduce((sum, p) => sum + p.share_pct, 0) / pts.length : 0;
  return Math.round((avg(second) - avg(first)) * 10) / 10;
}

function buildCategoryTrendSeries(
  window: StoredPrediction[],
  getLabel: (p: StoredPrediction) => string,
  knownLabels: string[],
): CategoryTrendSeries[] {
  const byDay = new Map<string, StoredPrediction[]>();
  for (const p of window) {
    const day = p.prediction_timestamp.slice(0, 10);
    const bucket = byDay.get(day) ?? [];
    bucket.push(p);
    byDay.set(day, bucket);
  }
  const days = [...byDay.keys()].sort();

  const activeLabels = [
    ...new Set([
      ...knownLabels,
      ...window.map((p) => getLabel(p)),
    ]),
  ].filter((label) => window.some((p) => getLabel(p) === label));

  return activeLabels
    .map((label) => {
      const points: CategoryTrendPoint[] = days.map((day) => {
        const rows = byDay.get(day) ?? [];
        const count = rows.filter((p) => getLabel(p) === label).length;
        const share = rows.length > 0 ? count / rows.length : 0;
        return {
          date: day,
          share_pct: Math.round(share * 1000) / 10,
          count,
        };
      });
      const total = window.filter((p) => getLabel(p) === label).length;
      return {
        label,
        display_label: label.replace(/_/g, " "),
        points,
        trend_pp: computeTrendPp(points),
        current_share_pct: Math.round((total / window.length) * 1000) / 10,
      };
    })
    .filter((s) => s.current_share_pct > 0 || s.points.some((p) => p.count > 0))
    .sort((a, b) => b.current_share_pct - a.current_share_pct);
}

function attachTrendsToShares(
  shares: CategoryShare[],
  trends: CategoryTrendSeries[],
): CategoryShare[] {
  const trendMap = new Map(trends.map((t) => [t.label, t.trend_pp]));
  return shares.map((row) => ({
    ...row,
    trend_pp: trendMap.get(row.label) ?? null,
  }));
}

const NUMERIC_REQUEST_FEATURES = [
  "surface_area",
  "number_of_rooms",
  "house_age",
  "energy_label_score",
  "dist_to_city_centre_km",
] as const;

const FEATURE_LABELS: Record<string, string> = {
  surface_area: "Surface area",
  number_of_rooms: "Number of rooms",
  house_age: "House age",
  energy_label_score: "Energy label score",
  dist_to_city_centre_km: "Distance to city centre",
};

function positionInTrainingBand(value: number, p01: number, p99: number): number {
  if (p99 <= p01) return 50;
  return ((value - p01) / (p99 - p01)) * 100;
}

function deterministicJitter(index: number): number {
  return 0.1 + ((index * 17) % 10) / 10;
}

function extractFeatureValues(
  window: StoredPrediction[],
  feature: string,
): { value: number; day: string }[] {
  const rows: { value: number; day: string }[] = [];
  for (const p of window) {
    const payload = parsePayload(p.request_payload);
    const value = Number(payload[feature]);
    if (!Number.isFinite(value)) continue;
    rows.push({ value, day: p.prediction_timestamp.slice(0, 10) });
  }
  return rows;
}

function buildDailyTrend(
  rows: { value: number; day: string }[],
  bounds: { p01: number; p99: number },
): FeatureDailyTrend[] {
  const byDay = new Map<string, { value: number; day: string }[]>();
  for (const row of rows) {
    const bucket = byDay.get(row.day) ?? [];
    bucket.push(row);
    byDay.set(row.day, bucket);
  }
  return [...byDay.entries()]
    .sort(([a], [b]) => a.localeCompare(b))
    .map(([date, dayRows]) => {
      const outside = dayRows.filter(
        (r) => r.value < bounds.p01 || r.value > bounds.p99,
      ).length;
      const mean = dayRows.reduce((a, r) => a + r.value, 0) / dayRows.length;
      return {
        date,
        mean: Math.round(mean * 100) / 100,
        pct_outside: Math.round((outside / dayRows.length) * 1000) / 10,
        n: dayRows.length,
      };
    });
}

function buildFeatureDistribution(
  feature: string,
  window: StoredPrediction[],
  manifest: TrainingManifest,
): FeatureDistributionViz | null {
  const bounds = manifest.feature_bounds[feature];
  if (!bounds) return null;

  const rows = extractFeatureValues(window, feature);
  if (rows.length === 0) return null;

  const values = rows.map((r) => r.value);
  const outside = values.filter((v) => v < bounds.p01 || v > bounds.p99).length;
  const recentMean = values.reduce((a, b) => a + b, 0) / values.length;
  const pctOut = (outside / values.length) * 100;

  const points: FeaturePointViz[] = rows.map((row, i) => ({
    index: i + 1,
    value: Math.round(row.value * 100) / 100,
    in_range: row.value >= bounds.p01 && row.value <= bounds.p99,
    position_pct: Math.round(positionInTrainingBand(row.value, bounds.p01, bounds.p99) * 10) / 10,
    jitter: Math.round(deterministicJitter(i) * 1000) / 1000,
    day: row.day,
  }));

  return {
    feature,
    label: FEATURE_LABELS[feature] ?? feature.replace(/_/g, " "),
    training_p01: bounds.p01,
    training_p99: bounds.p99,
    recent_mean: Math.round(recentMean * 100) / 100,
    pct_out_of_range: Math.round(pctOut * 10) / 10,
    sample_size: values.length,
    points,
    daily_trend: buildDailyTrend(rows, bounds),
  };
}

export function buildRequestMonitoring(
  predictions: StoredPrediction[],
  manifest: TrainingManifest,
  windowSize = 500,
): RequestMonitoring {
  const window = predictions.slice(0, windowSize);
  const warnings: string[] = [];

  if (window.length === 0) {
    return {
      sample_size: 0,
      window_label: `last ${windowSize} predictions`,
      by_region: [],
      by_property_type: [],
      region_trends: [],
      property_type_trends: [],
      numeric_features: [],
      feature_distributions: [],
      warnings: ["No prediction requests logged yet — run predictions via the app or API"],
    };
  }

  const regions = window.map((p) => p.region || "unknown");
  const propertyTypes = window.map((p) => p.property_type || "unknown");
  const regionTrends = buildCategoryTrendSeries(
    window,
    (p) => p.region || "unknown",
    manifest.regions,
  );
  const propertyTrends = buildCategoryTrendSeries(
    window,
    (p) => p.property_type || "unknown",
    manifest.property_types,
  );
  const byRegion = attachTrendsToShares(categoryShares(regions, manifest.regions), regionTrends);
  const byPropertyType = attachTrendsToShares(
    categoryShares(propertyTypes, manifest.property_types),
    propertyTrends,
  );

  for (const row of byRegion) {
    if (row.skew_pp != null && Math.abs(row.skew_pp) >= 15 && row.count > 0) {
      warnings.push(
        `Region skew: ${row.label} is ${row.skew_pp > 0 ? "+" : ""}${row.skew_pp.toFixed(1)}pp vs uniform training coverage`,
      );
    }
  }

  const numeric_features: FeatureDriftRow[] = [];
  const feature_distributions: FeatureDistributionViz[] = [];

  for (const feature of NUMERIC_REQUEST_FEATURES) {
    const dist = buildFeatureDistribution(feature, window, manifest);
    if (!dist) continue;

    feature_distributions.push(dist);
    numeric_features.push({
      feature,
      recent_mean: dist.recent_mean,
      reference_mean: 0,
      pct_out_of_range: dist.pct_out_of_range,
      sample_size: dist.sample_size,
    });

    if (dist.pct_out_of_range >= 10) {
      warnings.push(
        `${dist.label}: ${dist.pct_out_of_range.toFixed(1)}% of recent requests outside training p01–p99`,
      );
    }
  }

  return {
    sample_size: window.length,
    window_label: `last ${window.length} logged predictions`,
    by_region: byRegion,
    by_property_type: byPropertyType,
    region_trends: regionTrends,
    property_type_trends: propertyTrends,
    numeric_features,
    feature_distributions,
    warnings,
  };
}
