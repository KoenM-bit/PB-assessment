import { describe, expect, it } from "vitest";
import { buildRequestMonitoring } from "./request_monitoring.js";
import type { StoredPrediction } from "./databricks.js";
import type { TrainingManifest } from "./training_manifest.js";

function prediction(surface: number, rooms: number, day: string): StoredPrediction {
  return {
    prediction_id: "id",
    predicted_price: 1,
    model_name: "m",
    model_version: "1",
    model_alias: "challenger",
    prediction_timestamp: `${day}T12:00:00Z`,
    warnings: [],
    listing_id: null,
    address: "hidden",
    postcode: null,
    property_key: "k",
    request_payload: JSON.stringify({ surface_area: surface, number_of_rooms: rooms }),
    app_env: "staging",
    serving_latency_ms: 100,
    is_fallback: false,
    region: "Utrecht",
    property_type: "apartment",
    surface_area: surface,
  };
}

const manifest: TrainingManifest = {
  model_type: "rf",
  feature_pipeline_version: "1",
  training_date: "",
  git_commit: "",
  training_data_rows: 100,
  test_rows: 10,
  validation_approach: "holdout",
  regions: ["Utrecht"],
  property_types: ["apartment"],
  surface_area_range: { min: 50, max: 200, median: 100 },
  price_range: { min: 1, max: 2, median: 1.5 },
  feature_bounds: {
    surface_area: { p01: 50, p99: 200 },
    number_of_rooms: { p01: 2, p99: 8 },
  },
  baseline_lookup: {},
  global_median_psm: 3000,
  holdout_evaluation: {
    model: { mae: 1, rmse: 1, bias: 0, mape: 1, sample_size: 10 },
    baseline: { mae: 1, rmse: 1, bias: 0, mape: 1, sample_size: 10 },
    beats_baseline: true,
    mae_improvement_pct: 1,
  },
  walk_forward_baseline_mae_mean: null,
};

describe("buildRequestMonitoring feature distributions", () => {
  it("marks out-of-band values and builds normalized positions", () => {
    const preds = [
      prediction(100, 4, "2026-07-20"),
      prediction(250, 9, "2026-07-20"),
      prediction(60, 3, "2026-07-19"),
    ];
    const result = buildRequestMonitoring(preds, manifest, 50);
    const surface = result.feature_distributions.find((d) => d.feature === "surface_area");
    expect(surface?.points).toHaveLength(3);
    expect(surface?.points[0].in_range).toBe(true);
    expect(surface?.points[1].in_range).toBe(false);
    expect(surface?.points[1].position_pct).toBeGreaterThan(100);
    expect(surface?.daily_trend).toHaveLength(2);
  });
});
