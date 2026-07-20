import { describe, expect, it } from "vitest";
import { isBaselineFallback, isPeerServingFallback, type StoredPrediction } from "./databricks.js";

function prediction(overrides: Partial<StoredPrediction>): StoredPrediction {
  return {
    prediction_id: "id",
    predicted_price: 1,
    model_name: "house_price_model",
    model_version: "11",
    model_alias: "challenger",
    prediction_timestamp: "2026-07-20T10:00:00Z",
    warnings: [],
    listing_id: null,
    address: "a",
    postcode: null,
    property_key: "k",
    request_payload: "{}",
    app_env: "staging",
    serving_latency_ms: 100,
    is_fallback: false,
    region: "Utrecht",
    property_type: "apartment",
    surface_area: 80,
    ...overrides,
  };
}

describe("prediction fallback classification", () => {
  it("treats SQL string false as not fallback when is_fallback flag is false", () => {
    const row = prediction({ is_fallback: false });
    expect(isBaselineFallback(row)).toBe(false);
  });

  it("detects baseline fallback from model version and warnings", () => {
    expect(isBaselineFallback(prediction({ model_version: "baseline" }))).toBe(true);
    expect(
      isBaselineFallback(
        prediction({ warnings: ["fallback_to_business_baseline"], is_fallback: true }),
      ),
    ).toBe(true);
  });

  it("detects peer serving without counting as baseline", () => {
    const peer = prediction({
      model_version: "8",
      is_fallback: true,
      warnings: ["fallback_to_peer_serving:production"],
    });
    expect(isPeerServingFallback(peer)).toBe(true);
    expect(isBaselineFallback(peer)).toBe(false);
  });
});
