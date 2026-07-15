import { afterEach, describe, expect, it, vi } from "vitest";
import type { AppConfig } from "./config.js";
import { predictWithFallback } from "./fallback.js";

vi.mock("./databricks.js", () => ({
  invokeServing: vi.fn(),
}));

import { invokeServing } from "./databricks.js";

const baseConfig: AppConfig = {
  appEnv: "staging",
  databricksHost: "https://example.databricks.com",
  databricksToken: "token",
  servingEndpoint: "house-price-serving",
  sqlWarehouseId: "wh",
  catalog: "house_price_staging",
  schema: "gold",
  modelAlias: "challenger",
  peerServingTarget: {
    servingEndpoint: "house-price-serving-prod",
    catalog: "house_price_prod",
    modelAlias: "champion",
  },
  minEvaluationSampleSize: 30,
  demoWriteToken: "secret",
  servingTimeoutMs: 30000,
  sqlMaxWaitMs: 25000,
  useMockDatabricks: false,
};

const payload = {
  address: "Domstraat 12",
  postcode: "3512 JC",
  surface_area: 120,
  number_of_rooms: 5,
  number_of_bedrooms: 3,
  build_year: 1985,
  energy_label: "B" as const,
  property_type: "terraced_house" as const,
  garden: true,
  region: "Utrecht" as const,
  latitude: 52.0907,
  longitude: 5.1214,
};

describe("predictWithFallback", () => {
  afterEach(() => {
    vi.resetAllMocks();
  });

  it("uses primary serving when available", async () => {
    vi.mocked(invokeServing).mockResolvedValueOnce({
      predicted_price: 500000,
      model_version: "12",
      warnings: [],
    });

    const result = await predictWithFallback(baseConfig, payload, "challenger");
    expect(result.is_fallback).toBe(false);
    expect(result.model_alias).toBe("challenger");
    expect(invokeServing).toHaveBeenCalledTimes(1);
  });

  it("falls back to peer production serving when staging is unavailable", async () => {
    vi.mocked(invokeServing)
      .mockRejectedValueOnce(new Error("Serving failed: 503"))
      .mockResolvedValueOnce({
        predicted_price: 510000,
        model_version: "8",
        warnings: [],
      });

    const result = await predictWithFallback(baseConfig, payload, "challenger");
    expect(result.is_fallback).toBe(true);
    expect(result.model_alias).toBe("champion");
    expect(result.warnings).toContain("fallback_to_peer_serving:production");
    expect(invokeServing).toHaveBeenCalledTimes(2);
    expect(vi.mocked(invokeServing).mock.calls[1][0].servingEndpoint).toBe(
      "house-price-serving-prod",
    );
  });

  it("falls back to peer staging serving when production is unavailable", async () => {
    const prodConfig: AppConfig = {
      ...baseConfig,
      appEnv: "production",
      servingEndpoint: "house-price-serving-prod",
      catalog: "house_price_prod",
      modelAlias: "champion",
      peerServingTarget: {
        servingEndpoint: "house-price-serving",
        catalog: "house_price_staging",
        modelAlias: "challenger",
      },
    };

    vi.mocked(invokeServing)
      .mockRejectedValueOnce(new Error("timeout"))
      .mockResolvedValueOnce({
        predicted_price: 490000,
        model_version: "15",
        warnings: [],
      });

    const result = await predictWithFallback(prodConfig, payload, "champion");
    expect(result.is_fallback).toBe(true);
    expect(result.warnings).toContain("fallback_to_peer_serving:staging");
    expect(vi.mocked(invokeServing).mock.calls[1][0].servingEndpoint).toBe("house-price-serving");
  });

  it("uses business baseline when primary and peer serving fail", async () => {
    vi.mocked(invokeServing)
      .mockRejectedValueOnce(new Error("primary down"))
      .mockRejectedValueOnce(new Error("peer down"));

    const result = await predictWithFallback(baseConfig, payload, "challenger");
    expect(result.model_alias).toBe("baseline");
    expect(result.warnings).toContain("fallback_to_business_baseline");
    expect(result.is_fallback).toBe(true);
  });
});
