import type { AppConfig, ServingTarget } from "./config.js";
import type { PredictRequest } from "./schemas.js";
import { toModelPayload } from "./schemas.js";
import { invokeServing } from "./databricks.js";

const REGION_PSM: Record<string, Record<string, number>> = {
  Utrecht: { terraced_house: 4200, apartment: 4500, semi_detached: 4800, detached: 5200, bungalow: 4100 },
  Amsterdam: { terraced_house: 6500, apartment: 7000, semi_detached: 7500, detached: 8000, bungalow: 6000 },
};

function configForTarget(config: AppConfig, target: ServingTarget): AppConfig {
  return {
    ...config,
    servingEndpoint: target.servingEndpoint,
    catalog: target.catalog,
    modelAlias: target.modelAlias,
  };
}

function peerFallbackWarning(appEnv: string): string {
  return appEnv === "staging"
    ? "fallback_to_peer_serving:production"
    : "fallback_to_peer_serving:staging";
}

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
  const modelPayload = toModelPayload(payload);

  try {
    const result = await invokeServing(config, modelPayload, primaryAlias);
    return { ...result, model_alias: primaryAlias, is_fallback: false };
  } catch (err) {
    console.warn(`Primary serving failed (${config.servingEndpoint}):`, err);
  }

  if (config.peerServingTarget && !config.useMockDatabricks) {
    const peer = config.peerServingTarget;
    try {
      const result = await invokeServing(
        configForTarget(config, peer),
        modelPayload,
        peer.modelAlias,
      );
      return {
        ...result,
        model_alias: peer.modelAlias,
        is_fallback: true,
        warnings: [...result.warnings, peerFallbackWarning(config.appEnv)],
      };
    } catch (err) {
      console.warn(`Peer serving failed (${peer.servingEndpoint}):`, err);
    }
  }

  const psm = REGION_PSM[payload.region]?.[payload.property_type] ?? 3500;
  return {
    predicted_price: Math.round(psm * payload.surface_area),
    model_version: "baseline",
    model_alias: "baseline",
    warnings: ["fallback_to_business_baseline"],
    is_fallback: true,
  };
}
