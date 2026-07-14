import type { AppConfig } from "./config.js";
import type { PredictRequest } from "./schemas.js";
import { toModelPayload } from "./schemas.js";
import { invokeServing } from "./databricks.js";

const REGION_PSM: Record<string, Record<string, number>> = {
  Utrecht: { terraced_house: 4200, apartment: 4500, semi_detached: 4800, detached: 5200, bungalow: 4100 },
  Amsterdam: { terraced_house: 6500, apartment: 7000, semi_detached: 7500, detached: 8000, bungalow: 6000 },
};

export async function predictWithFallback(
  config: AppConfig,
  payload: PredictRequest,
  primaryAlias: string,
): Promise<{
  predicted_price: number;
  model_version: string;
  model_alias: string;
  warnings: string[];
  is_fallback: boolean;
}> {
  const aliases = [primaryAlias, "previous_champion"];
  const modelPayload = toModelPayload(payload);
  for (const alias of aliases) {
    try {
      const result = await invokeServing(config, modelPayload, alias);
      return { ...result, model_alias: alias, is_fallback: false };
    } catch (err) {
      console.warn(`Serving failed for alias ${alias}:`, err);
    }
  }

  // Business baseline fallback
  const psm = REGION_PSM[payload.region]?.[payload.property_type] ?? 3500;
  return {
    predicted_price: Math.round(psm * payload.surface_area),
    model_version: "baseline",
    model_alias: "baseline",
    warnings: ["fallback_to_business_baseline"],
    is_fallback: true,
  };
}
