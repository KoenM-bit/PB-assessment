import { describe, expect, it } from "vitest";
import { formatCurrency, formatPercent } from "../src/utils/format";

describe("format utilities", () => {
  it("formats currency", () => {
    expect(formatCurrency(438500)).toContain("438");
  });

  it("formats percent", () => {
    expect(formatPercent(5.5)).toBe("5.5%");
    expect(formatPercent(null)).toBe("—");
  });
});
