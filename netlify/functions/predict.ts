import type { Handler, HandlerEvent } from "@netlify/functions";
import { v4 as uuidv4 } from "uuid";
import { getConfig } from "./_shared/config.js";
import type { AppConfig } from "./_shared/config.js";
import {
  executeSqlWithRetry,
  insertPredictionSql,
  logPrediction,
  logServingEvent,
  newPredictionId,
  warmSqlWarehouse,
} from "./_shared/databricks.js";
import { ApiError, handleError, successResponse } from "./_shared/errors.js";
import { predictWithFallback } from "./_shared/fallback.js";
import { buildPropertyKey, predictRequestSchema } from "./_shared/schemas.js";

async function recordFailedRequest(
  config: AppConfig,
  err: unknown,
  latencyMs: number,
): Promise<void> {
  const apiErr =
    err instanceof ApiError
      ? err
      : err instanceof Error && err.name === "AbortError"
        ? new ApiError(504, "TIMEOUT", "Serving endpoint timed out")
        : new ApiError(500, "INTERNAL_ERROR", "An unexpected error occurred");

  await logServingEvent(config, {
    event_id: uuidv4(),
    event_timestamp: new Date().toISOString(),
    app_env: config.appEnv,
    http_status: apiErr.statusCode,
    error_code: apiErr.code,
    latency_ms: latencyMs,
    is_timeout: apiErr.code === "TIMEOUT" || apiErr.statusCode === 504,
  });
}

export const handler: Handler = async (event: HandlerEvent) => {
  const requestId = uuidv4();
  const start = Date.now();
  let config: AppConfig | null = null;

  try {
    if (event.httpMethod !== "POST") {
      throw new ApiError(405, "METHOD_NOT_ALLOWED", "POST required");
    }

    config = getConfig();

    const body = JSON.parse(event.body || "{}");
    const parsed = predictRequestSchema.safeParse(body);
    if (!parsed.success) {
      throw new ApiError(400, "VALIDATION_ERROR", parsed.error.message);
    }

    const warehouseWarmup = config.useMockDatabricks
      ? Promise.resolve()
      : warmSqlWarehouse(config).catch((err) => {
          console.warn("SQL warehouse warmup failed (will retry on insert):", err);
        });
    const result = await predictWithFallback(config, parsed.data, config.modelAlias);
    const latency = Date.now() - start;

    const predictionId = newPredictionId();
    const timestamp = new Date().toISOString();
    const record = {
      prediction_id: predictionId,
      predicted_price: result.predicted_price,
      model_name: "house_price_model",
      model_version: result.model_version,
      model_alias: result.model_alias,
      prediction_timestamp: timestamp,
      warnings: result.warnings,
      listing_id: parsed.data.listing_id ?? null,
      address: parsed.data.address,
      postcode: parsed.data.postcode ?? null,
      property_key: buildPropertyKey(parsed.data.address, parsed.data.postcode, parsed.data.region),
      request_payload: JSON.stringify(parsed.data),
      app_env: config.appEnv,
      serving_latency_ms: latency,
      is_fallback: result.is_fallback,
      region: parsed.data.region,
      property_type: parsed.data.property_type,
      surface_area: parsed.data.surface_area,
    };

    logPrediction(record);
    if (!config.useMockDatabricks) {
      await warehouseWarmup;
      try {
        await executeSqlWithRetry(config, insertPredictionSql(config, record));
      } catch (err) {
        console.error("Failed to persist prediction to Databricks:", err);
      }
    }

    return successResponse(
      {
        prediction_id: predictionId,
        predicted_price: result.predicted_price,
        model_name: "house_price_model",
        model_version: result.model_version,
        model_alias: result.model_alias,
        prediction_timestamp: timestamp,
        warnings: result.warnings,
        address: parsed.data.address,
        property_key: record.property_key,
      },
      requestId,
    );
  } catch (err) {
    const latency = Date.now() - start;
    if (config && !config.useMockDatabricks) {
      await recordFailedRequest(config, err, latency);
    }
    if (err instanceof Error && err.name === "AbortError") {
      return handleError(new ApiError(504, "TIMEOUT", "Serving endpoint timed out"), requestId);
    }
    return handleError(err, requestId);
  }
};
