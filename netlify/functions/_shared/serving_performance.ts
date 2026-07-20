import type { AppConfig } from "./config.js";
import type { StoredPrediction } from "./databricks.js";
import { isBaselineFallback } from "./databricks.js";

export interface LatencySummary {
  sample_size: number;
  avg_ms: number;
  p50_ms: number;
  p95_ms: number;
  max_ms: number;
}

export interface DailyServingPoint {
  date: string;
  request_count: number;
  p50_ms: number;
  p95_ms: number;
  fallback_count: number;
}

export interface DatabricksEndpointMetrics {
  endpoint_name: string;
  available: boolean;
  request_count_total: number;
  error_4xx_total: number;
  error_5xx_total: number;
  latency_p50_ms: number | null;
  latency_p99_ms: number | null;
  cpu_usage_pct: number | null;
  memory_usage_pct: number | null;
}

export interface ServingMetricsRow {
  date: string;
  request_count: number;
  error_count: number;
  timeout_count: number;
  p50_latency_ms: number;
  p95_latency_ms: number;
}

function percentile(sorted: number[], p: number): number {
  if (sorted.length === 0) return 0;
  const rank = Math.ceil((p / 100) * sorted.length) - 1;
  return sorted[Math.max(0, Math.min(rank, sorted.length - 1))];
}

export function summarizeLatencies(latenciesMs: number[]): LatencySummary {
  if (latenciesMs.length === 0) {
    return { sample_size: 0, avg_ms: 0, p50_ms: 0, p95_ms: 0, max_ms: 0 };
  }
  const sorted = [...latenciesMs].sort((a, b) => a - b);
  const sum = sorted.reduce((a, b) => a + b, 0);
  return {
    sample_size: sorted.length,
    avg_ms: Math.round(sum / sorted.length),
    p50_ms: Math.round(percentile(sorted, 50)),
    p95_ms: Math.round(percentile(sorted, 95)),
    max_ms: Math.round(sorted[sorted.length - 1]),
  };
}

export function dailyServingFromPredictions(predictions: StoredPrediction[]): DailyServingPoint[] {
  const byDay = new Map<string, StoredPrediction[]>();
  for (const p of predictions) {
    const day = p.prediction_timestamp.slice(0, 10);
    const bucket = byDay.get(day) ?? [];
    bucket.push(p);
    byDay.set(day, bucket);
  }

  return [...byDay.entries()]
    .sort(([a], [b]) => a.localeCompare(b))
    .map(([date, rows]) => {
      const latencies = rows.map((r) => r.serving_latency_ms);
      const summary = summarizeLatencies(latencies);
      return {
        date,
        request_count: rows.length,
        p50_ms: summary.p50_ms,
        p95_ms: summary.p95_ms,
        fallback_count: rows.filter((r) => isBaselineFallback(r)).length,
      };
    });
}

function parseMetricValue(raw: string): number {
  const n = Number(raw);
  return Number.isFinite(n) ? n : 0;
}

function parseLe(labelSegment: string): number | null {
  const match = labelSegment.match(/le="([^"]+)"/);
  if (!match) return null;
  if (match[1] === "+Inf") return Number.POSITIVE_INFINITY;
  const n = Number(match[1]);
  return Number.isFinite(n) ? n : null;
}

function histogramQuantile(
  buckets: { le: number; cumulative: number }[],
  quantile: number,
  total: number,
): number | null {
  if (total <= 0 || buckets.length === 0) return null;
  const target = total * quantile;
  const finite = buckets.filter((b) => Number.isFinite(b.le)).sort((a, b) => a.le - b.le);
  for (const bucket of finite) {
    if (bucket.cumulative >= target) return bucket.le;
  }
  const last = finite[finite.length - 1];
  return last ? last.le : null;
}

export function parseDatabricksServingMetrics(
  endpointName: string,
  body: string,
): DatabricksEndpointMetrics {
  const buckets: { le: number; cumulative: number }[] = [];
  let requestCount = 0;
  let error4xx = 0;
  let error5xx = 0;
  let cpu: number | null = null;
  let mem: number | null = null;
  let histogramCount = 0;

  for (const line of body.split("\n")) {
    const trimmed = line.trim();
    if (!trimmed || trimmed.startsWith("#")) continue;

    const gaugeMatch = trimmed.match(/^([a-zA-Z_:][\w:]*)\s+([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)/);
    if (gaugeMatch && !trimmed.includes("{")) {
      const name = gaugeMatch[1];
      const value = parseMetricValue(gaugeMatch[2]);
      if (name === "request_count_total") requestCount = value;
      if (name === "request_4xx_count_total") error4xx = value;
      if (name === "request_5xx_count_total") error5xx = value;
      if (name === "cpu_usage_percentage") cpu = value;
      if (name === "mem_usage_percentage") mem = value;
      if (name === "request_latency_ms_count") histogramCount = value;
      continue;
    }

    const labeledMatch = trimmed.match(
      /^([a-zA-Z_:][\w:]*)\{([^}]*)\}\s+([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)/,
    );
    if (!labeledMatch) continue;

    const metricName = labeledMatch[1];
    const labels = labeledMatch[2];
    const value = parseMetricValue(labeledMatch[3]);

    if (metricName === "request_latency_ms_bucket") {
      const le = parseLe(labels);
      if (le != null) buckets.push({ le, cumulative: value });
    }
  }

  const p50 = histogramQuantile(buckets, 0.5, histogramCount);
  const p99 = histogramQuantile(buckets, 0.99, histogramCount);

  return {
    endpoint_name: endpointName,
    available: true,
    request_count_total: Math.round(requestCount),
    error_4xx_total: Math.round(error4xx),
    error_5xx_total: Math.round(error5xx),
    latency_p50_ms: p50 != null && Number.isFinite(p50) ? Math.round(p50) : null,
    latency_p99_ms: p99 != null && Number.isFinite(p99) ? Math.round(p99) : null,
    cpu_usage_pct: cpu != null ? Math.round(cpu * 10) / 10 : null,
    memory_usage_pct: mem != null ? Math.round(mem * 10) / 10 : null,
  };
}

export async function fetchDatabricksEndpointMetrics(
  config: AppConfig,
): Promise<DatabricksEndpointMetrics | null> {
  if (config.useMockDatabricks || !config.databricksHost || !config.databricksToken) {
    return null;
  }

  const url = `${config.databricksHost}/api/2.0/serving-endpoints/${encodeURIComponent(config.servingEndpoint)}/metrics`;
  try {
    const response = await fetch(url, {
      headers: { Authorization: `Bearer ${config.databricksToken}` },
    });
    if (!response.ok) {
      console.warn(`Serving metrics API returned ${response.status} for ${config.servingEndpoint}`);
      return {
        endpoint_name: config.servingEndpoint,
        available: false,
        request_count_total: 0,
        error_4xx_total: 0,
        error_5xx_total: 0,
        latency_p50_ms: null,
        latency_p99_ms: null,
        cpu_usage_pct: null,
        memory_usage_pct: null,
      };
    }
    const text = await response.text();
    return parseDatabricksServingMetrics(config.servingEndpoint, text);
  } catch (err) {
    console.warn("Failed to fetch Databricks serving metrics:", err);
    return null;
  }
}
