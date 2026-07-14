import { describe, expect, it } from "vitest";
import { predictRequestSchema, actualSaleRequestSchema, buildPropertyKey } from "./schemas.js";

describe("predictRequestSchema", () => {
  const valid = {
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

  it("accepts valid input", () => {
    expect(predictRequestSchema.safeParse(valid).success).toBe(true);
  });

  it("rejects negative surface area", () => {
    expect(predictRequestSchema.safeParse({ ...valid, surface_area: -1 }).success).toBe(false);
  });

  it("rejects invalid energy label", () => {
    expect(predictRequestSchema.safeParse({ ...valid, energy_label: "Z" }).success).toBe(false);
  });

  it("rejects missing address", () => {
    const { address: _address, ...withoutAddress } = valid;
    expect(predictRequestSchema.safeParse(withoutAddress).success).toBe(false);
  });

  it("builds a stable property key", () => {
    expect(buildPropertyKey("Domstraat 12", "3512 JC", "Utrecht")).toBe(
      "domstraat 12|3512 jc|utrecht",
    );
  });
});

describe("actualSaleRequestSchema", () => {
  it("accepts valid actual sale", () => {
    const result = actualSaleRequestSchema.safeParse({
      prediction_id: "550e8400-e29b-41d4-a716-446655440000",
      actual_sale_price: 425000,
      sale_date: "2026-08-01",
    });
    expect(result.success).toBe(true);
  });
});
