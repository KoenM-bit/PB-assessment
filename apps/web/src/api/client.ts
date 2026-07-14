import type {
  ApiEnvelope,
  MonitoringData,
  PredictRequest,
  PredictionListItem,
  PredictionResult,
} from "../types";

const API_BASE = import.meta.env.VITE_API_BASE || "/api";

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    headers: { "Content-Type": "application/json", ...options?.headers },
    ...options,
  });

  const text = await response.text();
  let body: ApiEnvelope<T>;
  try {
    body = JSON.parse(text) as ApiEnvelope<T>;
  } catch {
    throw new Error(
      text.includes("Function not found")
        ? "API unavailable. Run `make dev-full` from the project root (not `make dev` alone)."
        : text.toLowerCase().includes("inactivity timeout")
          ? "Databricks timed out (warehouse or model endpoint waking up). Wait a moment and try again."
          : `Invalid API response: ${text.slice(0, 80)}`,
    );
  }

  if (!response.ok || body.error) {
    throw new Error(body.error?.message || `Request failed: ${response.status}`);
  }
  return body.data as T;
}

export const api = {
  predict: (data: PredictRequest) =>
    request<PredictionResult>("/predict", { method: "POST", body: JSON.stringify(data) }),

  getPredictions: (limit = 50) =>
    request<{ items: PredictionListItem[]; total: number }>(`/predictions?limit=${limit}`),

  submitActualSale: (data: { prediction_id: string; actual_sale_price: number; sale_date: string }, token: string) =>
    request<unknown>("/actual-sales", {
      method: "POST",
      headers: { Authorization: `Bearer ${token}` },
      body: JSON.stringify(data),
    }),

  getMonitoring: () => request<MonitoringData>("/monitoring"),
};
