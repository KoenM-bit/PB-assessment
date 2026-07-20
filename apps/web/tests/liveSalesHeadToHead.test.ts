import { describe, expect, it } from "vitest";
import { computeLiveSalesHeadToHead } from "../src/utils/liveSalesHeadToHead";
import type { LiveLabelledSale } from "../src/types";

function row(overrides: Partial<LiveLabelledSale> & Pick<LiveLabelledSale, "model_abs_error" | "baseline_abs_error">): LiveLabelledSale {
  return {
    prediction_id: "x",
    address: "A",
    region: "Utrecht",
    property_type: "apartment",
    surface_area: 80,
    predicted_price: 400_000,
    baseline_price: 410_000,
    actual_sale_price: 405_000,
    model_pct_error: 1,
    prediction_date: "2026-01-01",
    sale_date: null,
    beats_baseline: overrides.model_abs_error < overrides.baseline_abs_error,
    ...overrides,
  };
}

describe("computeLiveSalesHeadToHead", () => {
  it("computes win rates and average euro deltas", () => {
    const stats = computeLiveSalesHeadToHead([
      row({ model_abs_error: 10_000, baseline_abs_error: 28_400 }),
      row({ model_abs_error: 20_000, baseline_abs_error: 26_700 }),
      row({ model_abs_error: 30_000, baseline_abs_error: 15_000 }),
    ]);
    expect(stats.model_wins).toBe(2);
    expect(stats.baseline_wins).toBe(1);
    expect(stats.model_better_pct).toBeCloseTo(66.7, 0);
    expect(stats.avg_win_when_model_better_eur).toBeCloseTo(12_550, 0);
    expect(stats.avg_loss_when_baseline_better_eur).toBeCloseTo(15_000, 0);
  });
});
