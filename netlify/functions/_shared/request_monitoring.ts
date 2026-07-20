import type { StoredPrediction } from "./databricks.js";
import type { TrainingManifest } from "./training_manifest.js";

export interface CategoryShare {
  label: string;
  count: number;
  share: number;
  expected_share: number | null;
  skew_pp: number | null;
}

export interface FeatureDriftRow {
  feature: string;
  recent_mean: number;
  reference_mean: number;
  pct_out_of_range: number;
  sample_size: number;
}

export interface RequestMonitoring {
  sample_size: number;
  window_label: string;
  by_region: CategoryShare[];
  by_property_type: CategoryShare[];
  numeric_features: FeatureDriftRow[];
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
    };
  });
}

const NUMERIC_REQUEST_FEATURES = [
  "surface_area",
  "number_of_rooms",
  "house_age",
  "energy_label_score",
  "dist_to_city_centre_km",
] as const;

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
      numeric_features: [],
      warnings: ["No prediction requests logged yet — run predictions via the app or API"],
    };
  }

  const regions = window.map((p) => p.region || "unknown");
  const propertyTypes = window.map((p) => p.property_type || "unknown");
  const byRegion = categoryShares(regions, manifest.regions);
  const byPropertyType = categoryShares(propertyTypes, manifest.property_types);

  for (const row of byRegion) {
    if (row.skew_pp != null && Math.abs(row.skew_pp) >= 15 && row.count > 0) {
      warnings.push(
        `Region skew: ${row.label} is ${row.skew_pp > 0 ? "+" : ""}${row.skew_pp.toFixed(1)}pp vs uniform training coverage`,
      );
    }
  }

  const numeric_features: FeatureDriftRow[] = [];
  for (const feature of NUMERIC_REQUEST_FEATURES) {
    const bounds = manifest.feature_bounds[feature];
    if (!bounds) continue;

    const values: number[] = [];
    let outside = 0;
    for (const p of window) {
      const payload = parsePayload(p.request_payload);
      const raw = payload[feature];
      const value = Number(raw);
      if (!Number.isFinite(value)) continue;
      values.push(value);
      if (value < bounds.p01 || value > bounds.p99) outside += 1;
    }

    if (values.length === 0) continue;
    const recentMean = values.reduce((a, b) => a + b, 0) / values.length;
    const pctOut = (outside / values.length) * 100;
    numeric_features.push({
      feature,
      recent_mean: Math.round(recentMean * 100) / 100,
      reference_mean: 0,
      pct_out_of_range: Math.round(pctOut * 10) / 10,
      sample_size: values.length,
    });

    if (pctOut >= 10) {
      warnings.push(
        `${feature.replace(/_/g, " ")}: ${pctOut.toFixed(1)}% of recent requests outside training p01–p99`,
      );
    }
  }

  return {
    sample_size: window.length,
    window_label: `last ${window.length} logged predictions`,
    by_region: byRegion,
    by_property_type: byPropertyType,
    numeric_features,
    warnings,
  };
}
