import type { Handler, HandlerEvent } from "@netlify/functions";
import { v4 as uuidv4 } from "uuid";
import { getConfig } from "./_shared/config.js";
import {
  executeSql,
  getPredictionById,
  insertActualSaleSql,
  logActualSale,
} from "./_shared/databricks.js";
import { ApiError, handleError, successResponse } from "./_shared/errors.js";
import { actualSaleRequestSchema } from "./_shared/schemas.js";

export const handler: Handler = async (event: HandlerEvent) => {
  const requestId = uuidv4();
  try {
    if (event.httpMethod !== "POST") {
      throw new ApiError(405, "METHOD_NOT_ALLOWED", "POST required");
    }

    const config = getConfig();
    const auth = event.headers.authorization || "";
    const token = auth.replace("Bearer ", "");
    if (config.demoWriteToken && token !== config.demoWriteToken) {
      throw new ApiError(401, "UNAUTHORIZED", "Invalid or missing write token");
    }

    const body = JSON.parse(event.body || "{}");
    const parsed = actualSaleRequestSchema.safeParse(body);
    if (!parsed.success) {
      throw new ApiError(400, "VALIDATION_ERROR", parsed.error.message);
    }

    const prediction = await getPredictionById(config, parsed.data.prediction_id);
    if (!prediction) {
      throw new ApiError(404, "NOT_FOUND", "Prediction not found");
    }

    const record = {
      actual_sale_id: uuidv4(),
      prediction_id: parsed.data.prediction_id,
      listing_id: prediction.listing_id,
      actual_sale_price: parsed.data.actual_sale_price,
      sale_date: parsed.data.sale_date,
      recorded_at: new Date().toISOString(),
      recorded_by: "demo_user",
    };

    logActualSale(record);

    if (!config.useMockDatabricks) {
      await executeSql(config, insertActualSaleSql(config, record));
    }

    return successResponse(record, requestId, 201);
  } catch (err) {
    return handleError(err, requestId);
  }
};
