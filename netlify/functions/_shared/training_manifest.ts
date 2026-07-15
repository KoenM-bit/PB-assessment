import manifestJson from "./training_manifest.json" with { type: "json" };

export interface MetricSet {
  mae: number;
  rmse: number;
  bias: number;
  mape: number;
  sample_size: number;
}

export interface TrainingManifest {
  model_type: string;
  feature_pipeline_version: string;
  training_date: string;
  git_commit: string;
  training_data_rows: number;
  test_rows: number;
  validation_approach: string;
  regions: string[];
  property_types: string[];
  surface_area_range: { min: number; max: number; median: number };
  price_range: { min: number; max: number; median: number };
  feature_bounds: Record<string, { p01: number; p99: number }>;
  baseline_lookup: Record<string, number>;
  global_median_psm: number;
  holdout_evaluation: {
    model: MetricSet;
    baseline: MetricSet;
    beats_baseline: boolean;
    mae_improvement_pct: number;
  };
  walk_forward_baseline_mae_mean: number | null;
  walk_forward_model_mae_mean?: number | null;
  gates_passed?: boolean;
  gate_failures?: string[];
}

const DEFAULT_MANIFEST: TrainingManifest = {
  model_type: "random_forest",
  feature_pipeline_version: "1.0.0",
  training_date: "unknown",
  git_commit: "unknown",
  training_data_rows: 0,
  test_rows: 0,
  validation_approach: "walk_forward + holdout_test",
  regions: [],
  property_types: [],
  surface_area_range: { min: 0, max: 0, median: 0 },
  price_range: { min: 0, max: 0, median: 0 },
  feature_bounds: {},
  baseline_lookup: {},
  global_median_psm: 3000,
  holdout_evaluation: {
    model: { mae: 0, rmse: 0, bias: 0, mape: 0, sample_size: 0 },
    baseline: { mae: 0, rmse: 0, bias: 0, mape: 0, sample_size: 0 },
    beats_baseline: false,
    mae_improvement_pct: 0,
  },
  walk_forward_baseline_mae_mean: null,
};

let cachedManifest: TrainingManifest | null = null;

export function loadTrainingManifest(): TrainingManifest {
  if (cachedManifest) return cachedManifest;
  cachedManifest = manifestJson as TrainingManifest;
  return cachedManifest;
}

export function baselinePredict(
  region: string,
  propertyType: string,
  surfaceArea: number,
  manifest: TrainingManifest,
): number {
  const psm = manifest.baseline_lookup[`${region}|${propertyType}`] ?? manifest.global_median_psm;
  return psm * surfaceArea;
}
