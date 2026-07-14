import { v4 as uuidv4 } from "uuid";
import type { AppConfig } from "./config.js";
import type { ModelPredictPayload, PredictRequest, PredictionResponse } from "./schemas.js";

export interface StoredPrediction extends PredictionResponse {
  listing_id: string | null;
  address: string;
  postcode: string | null;
  property_key: string;
  request_payload: string;
  app_env: string;
  serving_latency_ms: number;
  is_fallback: boolean;
  region: string;
  property_type: string;
  surface_area: number;
}

export interface StoredActualSale {
  actual_sale_id: string;
  prediction_id: string;
  listing_id: string | null;
  actual_sale_price: number;
  sale_date: string;
  recorded_at: string;
  recorded_by: string;
}

type SqlStatementResult = {
  status?: { state?: string; error?: { message?: string } };
  result?: { data_array?: unknown[][] };
  manifest?: { schema?: { columns?: { name: string }[] } };
};

// In-memory store for local/mock mode
const predictions: StoredPrediction[] = [];
const actualSales: StoredActualSale[] = [];

function escapeSql(value: string): string {
  return value.replace(/'/g, "''");
}

export function logPrediction(record: StoredPrediction): void {
  predictions.unshift(record);
}

export function logActualSale(record: StoredActualSale): void {
  actualSales.push(record);
}

export async function getPredictions(
  config: AppConfig,
  limit = 50,
  offset = 0,
): Promise<StoredPrediction[]> {
  if (config.useMockDatabricks) {
    return predictions.slice(offset, offset + limit);
  }

  const rows = await querySql(
    config,
    `SELECT prediction_id, listing_id, address, postcode, property_key, request_payload,
            predicted_price, model_name, model_version, model_alias, prediction_timestamp,
            app_env, serving_latency_ms, warnings, is_fallback
     FROM ${config.catalog}.gold.predictions
     ORDER BY prediction_timestamp DESC
     LIMIT ${limit} OFFSET ${offset}`,
  );
  return rows.map(parsePredictionRow);
}

export async function getPredictionById(
  config: AppConfig,
  id: string,
): Promise<StoredPrediction | undefined> {
  if (config.useMockDatabricks) {
    return predictions.find((p) => p.prediction_id === id);
  }

  const rows = await querySql(
    config,
    `SELECT prediction_id, listing_id, address, postcode, property_key, request_payload,
            predicted_price, model_name, model_version, model_alias, prediction_timestamp,
            app_env, serving_latency_ms, warnings, is_fallback
     FROM ${config.catalog}.gold.predictions
     WHERE prediction_id = '${escapeSql(id)}'
     LIMIT 1`,
  );
  return rows[0] ? parsePredictionRow(rows[0]) : undefined;
}

export async function getActualSales(config: AppConfig): Promise<StoredActualSale[]> {
  if (config.useMockDatabricks) {
    return actualSales;
  }

  const rows = await querySql(
    config,
    `SELECT actual_sale_id, prediction_id, listing_id, actual_sale_price, sale_date,
            recorded_at, recorded_by
     FROM ${config.catalog}.gold.actual_sales
     ORDER BY recorded_at DESC`,
  );
  return rows.map(parseActualSaleRow);
}

function parsePredictionRow(row: Record<string, unknown>): StoredPrediction {
  const payload = parseRequestPayload(String(row.request_payload ?? "{}"));
  const warnings = parseWarnings(row.warnings);

  return {
    prediction_id: String(row.prediction_id),
    listing_id: row.listing_id ? String(row.listing_id) : null,
    address: String(row.address ?? payload.address ?? ""),
    postcode: row.postcode ? String(row.postcode) : payload.postcode ?? null,
    property_key: String(row.property_key ?? ""),
    request_payload: String(row.request_payload ?? "{}"),
    predicted_price: Number(row.predicted_price),
    model_name: String(row.model_name ?? "house_price_model"),
    model_version: String(row.model_version ?? "unknown"),
    model_alias: String(row.model_alias ?? "unknown"),
    prediction_timestamp: String(row.prediction_timestamp),
    app_env: String(row.app_env ?? ""),
    serving_latency_ms: Number(row.serving_latency_ms ?? 0),
    warnings,
    is_fallback: Boolean(row.is_fallback),
    region: String(payload.region ?? ""),
    property_type: String(payload.property_type ?? ""),
    surface_area: Number(payload.surface_area ?? 0),
  };
}

function parseActualSaleRow(row: Record<string, unknown>): StoredActualSale {
  return {
    actual_sale_id: String(row.actual_sale_id),
    prediction_id: String(row.prediction_id),
    listing_id: row.listing_id ? String(row.listing_id) : null,
    actual_sale_price: Number(row.actual_sale_price),
    sale_date: String(row.sale_date).slice(0, 10),
    recorded_at: String(row.recorded_at),
    recorded_by: String(row.recorded_by ?? ""),
  };
}

function parseRequestPayload(raw: string): Partial<PredictRequest> {
  try {
    return JSON.parse(raw) as Partial<PredictRequest>;
  } catch {
    return {};
  }
}

function parseWarnings(raw: unknown): string[] {
  if (Array.isArray(raw)) {
    return raw.map(String);
  }
  if (typeof raw === "string" && raw.trim()) {
    try {
      const parsed = JSON.parse(raw);
      return Array.isArray(parsed) ? parsed.map(String) : [raw];
    } catch {
      return [raw];
    }
  }
  return [];
}

export async function invokeServing(
  config: AppConfig,
  payload: ModelPredictPayload,
  alias: string,
): Promise<{ predicted_price: number; model_version: string; warnings: string[] }> {
  if (config.useMockDatabricks || !config.databricksHost || !config.databricksToken) {
    return mockPredict(payload);
  }

  const url = `${config.databricksHost}/serving-endpoints/${config.servingEndpoint}/invocations`;
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), config.servingTimeoutMs);

  try {
    const response = await fetch(url, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${config.databricksToken}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        dataframe_records: [payload],
      }),
      signal: controller.signal,
    });

    if (!response.ok) {
      throw new Error(`Serving failed: ${response.status}`);
    }

    const result = await response.json();
    const predictionRows = result.predictions || result;
    const row = Array.isArray(predictionRows) ? predictionRows[0] : predictionRows;
    const price = row?.predicted_price ?? row;
    const modelVersion =
      result.model_version ||
      result.databricks_model_version ||
      (await resolveDeployedModelVersion(config)) ||
      "unknown";

    return {
      predicted_price: Number(price),
      model_version: String(modelVersion),
      warnings: parsePredictionWarnings(row?.warnings),
    };
  } finally {
    clearTimeout(timeout);
  }
}

function parsePredictionWarnings(raw: unknown): string[] {
  if (Array.isArray(raw)) {
    return raw.map(String);
  }
  if (typeof raw === "string" && raw.trim()) {
    try {
      const parsed = JSON.parse(raw);
      return Array.isArray(parsed) ? parsed.map(String) : [raw];
    } catch {
      return [raw];
    }
  }
  return [];
}

async function resolveDeployedModelVersion(config: AppConfig): Promise<string | null> {
  if (!config.databricksHost || !config.databricksToken || !config.servingEndpoint) {
    return null;
  }

  const response = await fetch(
    `${config.databricksHost}/api/2.0/serving-endpoints/${config.servingEndpoint}`,
    {
      headers: { Authorization: `Bearer ${config.databricksToken}` },
    },
  );
  if (!response.ok) {
    return null;
  }

  const endpoint = await response.json();
  const entities =
    endpoint.config?.served_entities ||
    endpoint.pending_config?.served_entities ||
    [];
  const version = entities[0]?.entity_version;
  return version != null ? String(version) : null;
}

function mockPredict(
  payload: ModelPredictPayload,
): { predicted_price: number; model_version: string; warnings: string[] } {
  const regionFactor: Record<string, number> = {
    Amsterdam: 1.35,
    Rotterdam: 1.05,
    Utrecht: 1.25,
    "The Hague": 1.15,
    Eindhoven: 0.95,
    Groningen: 0.85,
    Maastricht: 0.9,
    Nijmegen: 0.88,
  };
  const base = (regionFactor[payload.region] || 1) * payload.surface_area * 3200;
  const warnings: string[] = [];
  if (payload.surface_area > 200) warnings.push("surface_area above training p99");
  return {
    predicted_price: Math.round(base),
    model_version: "mock-v1",
    warnings,
  };
}

export async function executeSql(config: AppConfig, statement: string): Promise<unknown> {
  if (config.useMockDatabricks) return null;
  const url = `${config.databricksHost}/api/2.0/sql/statements`;
  const response = await fetch(url, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${config.databricksToken}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      warehouse_id: config.sqlWarehouseId,
      statement,
      wait_timeout: "30s",
    }),
  });
  if (!response.ok) throw new Error(`SQL failed: ${response.status}`);
  const result = (await response.json()) as SqlStatementResult;
  if (result.status?.state === "FAILED") {
    throw new Error(result.status.error?.message || "SQL statement failed");
  }
  return result;
}

export async function querySql(
  config: AppConfig,
  statement: string,
): Promise<Record<string, unknown>[]> {
  const result = (await executeSql(config, statement)) as SqlStatementResult;
  const columns = result.manifest?.schema?.columns?.map((column) => column.name) ?? [];
  const rows = result.result?.data_array ?? [];
  return rows.map((row) =>
    Object.fromEntries(columns.map((name, index) => [name, row[index]])),
  );
}

export function insertPredictionSql(config: AppConfig, record: StoredPrediction): string {
  const warnings = record.warnings.map((w) => `'${escapeSql(w)}'`).join(",") || "";
  return `INSERT INTO ${config.catalog}.gold.predictions (
    prediction_id,
    listing_id,
    address,
    postcode,
    property_key,
    request_payload,
    predicted_price,
    model_name,
    model_version,
    model_alias,
    prediction_timestamp,
    app_env,
    serving_latency_ms,
    warnings,
    is_fallback
  ) VALUES (
    '${escapeSql(record.prediction_id)}',
    ${record.listing_id ? `'${escapeSql(record.listing_id)}'` : "NULL"},
    '${escapeSql(record.address)}',
    ${record.postcode ? `'${escapeSql(record.postcode)}'` : "NULL"},
    '${escapeSql(record.property_key)}',
    '${escapeSql(record.request_payload)}',
    ${record.predicted_price},
    '${escapeSql(record.model_name)}',
    '${escapeSql(record.model_version)}',
    '${escapeSql(record.model_alias)}',
    '${escapeSql(record.prediction_timestamp)}',
    '${escapeSql(record.app_env)}',
    ${record.serving_latency_ms},
    array(${warnings}),
    ${record.is_fallback}
  )`;
}

export function insertActualSaleSql(config: AppConfig, record: StoredActualSale): string {
  return `INSERT INTO ${config.catalog}.gold.actual_sales (
    actual_sale_id,
    prediction_id,
    listing_id,
    actual_sale_price,
    sale_date,
    recorded_at,
    recorded_by
  ) VALUES (
    '${escapeSql(record.actual_sale_id)}',
    '${escapeSql(record.prediction_id)}',
    ${record.listing_id ? `'${escapeSql(record.listing_id)}'` : "NULL"},
    ${record.actual_sale_price},
    '${escapeSql(record.sale_date)}',
    '${escapeSql(record.recorded_at)}',
    '${escapeSql(record.recorded_by)}'
  )`;
}

export function newPredictionId(): string {
  return uuidv4();
}
