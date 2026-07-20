import { describe, expect, it } from "vitest";
import { parseDatabricksServingMetrics } from "./serving_performance.js";

describe("parseDatabricksServingMetrics", () => {
  it("parses labeled OpenMetrics series (replica labels)", () => {
    const body = `
# HELP request_count_total Total requests
request_count_total{replica_id="0"} 42
request_count_total{replica_id="1"} 37
cpu_usage_percentage{replica_id="0"} 20
cpu_usage_percentage{replica_id="1"} 30
request_latency_ms_count{replica_id="0"} 42
request_latency_ms_bucket{le="100",replica_id="0"} 40
request_latency_ms_bucket{le="250",replica_id="0"} 42
request_latency_ms_bucket{le="+Inf",replica_id="0"} 42
`.trim();

    const result = parseDatabricksServingMetrics("house-price-serving", body);

    expect(result.has_metrics).toBe(true);
    expect(result.request_count_total).toBe(79);
    expect(result.cpu_usage_pct).toBe(25);
    expect(result.latency_p50_ms).toBe(100);
  });
});
