import type { Handler, HandlerEvent } from "@netlify/functions";
import { v4 as uuidv4 } from "uuid";
import { getConfig } from "./_shared/config.js";
import { getActualSales, getPredictions } from "./_shared/databricks.js";
import { handleError, successResponse } from "./_shared/errors.js";

export const handler: Handler = async (event: HandlerEvent) => {
  const requestId = uuidv4();
  try {
    const config = getConfig();
    const limit = parseInt(event.queryStringParameters?.limit || "50", 10);
    const offset = parseInt(event.queryStringParameters?.offset || "0", 10);
    const predictions = await getPredictions(config, limit, offset);
    const actuals = await getActualSales(config);
    const actualMap = new Map(actuals.map((a) => [a.prediction_id, a]));

    const items = predictions.map((p) => {
      const actual = actualMap.get(p.prediction_id);
      const absError = actual
        ? Math.abs(actual.actual_sale_price - p.predicted_price)
        : null;
      const pctError = actual
        ? (absError! / actual.actual_sale_price) * 100
        : null;
      return {
        prediction_id: p.prediction_id,
        listing_id: p.listing_id,
        address: p.address,
        postcode: p.postcode,
        property_key: p.property_key,
        region: p.region,
        property_type: p.property_type,
        surface_area: p.surface_area,
        predicted_price: p.predicted_price,
        actual_sale_price: actual?.actual_sale_price ?? null,
        absolute_error: absError,
        percentage_error: pctError,
        model_version: p.model_version,
        prediction_date: p.prediction_timestamp,
        sale_date: actual?.sale_date ?? null,
      };
    });

    return successResponse({ items, total: items.length }, requestId);
  } catch (err) {
    return handleError(err, requestId);
  }
};
