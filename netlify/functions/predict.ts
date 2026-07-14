import type { Handler, HandlerEvent } from "@netlify/functions";
import { v4 as uuidv4 } from "uuid";
import { getConfig } from "./_shared/config.js";
import {
  executeSql,
  insertPredictionSql,
  logPrediction,
  newPredictionId,
} from "./_shared/databricks.js";
import { ApiError, handleError, successResponse } from "./_shared/errors.js";
import { predictWithFallback } from "./_shared/fallback.js";
import { buildPropertyKey, predictRequestSchema } from "./_shared/schemas.js";

export const handler: Handler = async (event: HandlerEvent) => {
  const requestId = uuidv4();
  try {
    if (event.httpMethod !== "POST") {
      throw new ApiError(405, "METHOD_NOT_ALLOWED", "POST required");
    }

    const body = JSON.parse(event.body || "{}");
    const parsed = predictRequestSchema.safeParse(body);
    if (!parsed.success) {
      throw new ApiError(400, "VALIDATION_ERROR", parsed.error.message);
    }

    const config = getConfig();
    const start = Date.now();
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
      await executeSql(config, insertPredictionSql(config, record));
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
    if (err instanceof Error && err.name === "AbortError") {
      return handleError(new ApiError(504, "TIMEOUT", "Serving endpoint timed out"), requestId);
    }
    return handleError(err, requestId);
  }
};
