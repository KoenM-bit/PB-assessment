import { describe, expect, it } from "vitest";
import { buildLiveLabelledMonitoring } from "./live_evaluation_monitoring.js";

describe("buildLiveLabelledMonitoring", () => {
  it("builds segment bars, trends, and labelled items", () => {
    const rows = [
      {
        prediction_id: "a",
        address: "A 1",
        region: "Utrecht",
        property_type: "apartment",
        surface_area: 80,
        predicted: 400_000,
        actual: 410_000,
        baseline: 420_000,
        prediction_timestamp: "2026-07-10T12:00:00Z",
        sale_date: "2026-07-20",
      },
      {
        prediction_id: "b",
        address: "B 2",
        region: "Utrecht",
        property_type: "apartment",
        surface_area: 90,
        predicted: 450_000,
        actual: 440_000,
        baseline: 460_000,
        prediction_timestamp: "2026-07-18T12:00:00Z",
        sale_date: "2026-07-25",
      },
      {
        prediction_id: "c",
        address: "C 3",
        region: "Amsterdam",
        property_type: "terraced_house",
        surface_area: 100,
        predicted: 600_000,
        actual: 700_000,
        baseline: 650_000,
        prediction_timestamp: "2026-07-18T14:00:00Z",
        sale_date: null,
      },
    ];

    const out = buildLiveLabelledMonitoring(rows);
    expect(out.sample_size).toBe(3);
    expect(out.by_region.find((r) => r.label === "Utrecht")?.sample_size).toBe(2);
    expect(out.items[0].prediction_id).toBe("c");
    expect(out.items.find((i) => i.prediction_id === "a")?.beats_baseline).toBe(false);
    expect(out.region_mae_trends.some((t) => t.label === "Utrecht")).toBe(true);
  });
});
