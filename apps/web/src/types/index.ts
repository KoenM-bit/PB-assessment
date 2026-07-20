export interface PredictRequest {
  address: string;
  postcode?: string;
  listing_id?: string;
  surface_area: number;
  number_of_rooms: number;
  number_of_bedrooms: number;
  build_year: number;
  energy_label: string;
  property_type: string;
  garden: boolean;
  region: string;
  latitude: number;
  longitude: number;
  prediction_date?: string;
}

export interface PredictionResult {
  prediction_id: string;
  predicted_price: number;
  model_name: string;
  model_version: string;
  model_alias: string;
  prediction_timestamp: string;
  warnings: string[];
  address: string;
  property_key: string;
}

export interface PredictionListItem {
  prediction_id: string;
  listing_id: string | null;
  address: string;
  postcode: string | null;
  property_key: string;
  region: string;
  property_type: string;
  surface_area: number;
  predicted_price: number;
  actual_sale_price: number | null;
  absolute_error: number | null;
  percentage_error: number | null;
  model_version: string;
  prediction_date: string;
  sale_date: string | null;
}

export interface MetricSet {
  mae: number;
  rmse: number;
  bias: number;
  mape: number;
  sample_size: number;
}

export interface ModelComparison {
  model: MetricSet;
  baseline: MetricSet;
  beats_baseline: boolean;
  mae_improvement_pct: number;
  is_reliable?: boolean;
}

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

export interface ServingHistoryPoint {
  date: string;
  request_count: number;
  p50_ms: number;
  p95_ms: number;
  error_count: number;
  timeout_count: number;
}

export interface MonitoringData {
  summary: {
    total_predictions: number;
    labelled_predictions: number;
    active_model_version: string;
    min_sample_size: number;
  };
  training: {
    model_type: string;
    feature_pipeline_version: string;
    training_date: string;
    git_commit: string;
    training_data_rows: number;
    test_rows: number;
    validation_approach: string;
    regions: string[];
    property_types: string[];
    surface_area_range: { min: number; max: number; median: number };
    price_range: { min: number; max: number; median: number };
    feature_bounds_summary: { feature: string; p01: number; p99: number }[];
    walk_forward_baseline_mae_mean: number | null;
  };
  holdout_evaluation: ModelComparison;
  live_evaluation: {
    overall: ModelComparison;
    by_region: Record<string, ModelComparison>;
  };
  performance: {
    overall: { mae: number; rmse: number; bias: number; sample_size: number; is_reliable: boolean };
    by_region: Record<string, { mae: number; rmse: number; bias: number; sample_size: number }>;
    by_property_type: Record<string, { mae: number; rmse: number; bias: number; sample_size: number }>;
  };
  infrastructure: {
    request_count: number;
    error_rate: number;
    timeout_rate: number;
    api_latency: LatencySummary;
    fallback_rate: number;
    daily: DailyServingPoint[];
    history: ServingHistoryPoint[];
    databricks_endpoint: DatabricksEndpointMetrics | null;
    serving_endpoint: string;
  };
  data_quality: { missing_value_rate: number; out_of_range_rate: number };
  prediction_distribution: { mean: number; count: number };
  warnings: string[];
}

export interface ApiEnvelope<T> {
  data: T | null;
  error: { code: string; message: string } | null;
  meta: { request_id: string; timestamp: string };
}
