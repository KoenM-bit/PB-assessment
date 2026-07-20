# Monitoring

## Monitoring Layers

### 1. Infrastructure & API

| Metric | Source |
|--------|--------|
| Request count | `gold.serving_metrics`, Netlify logs |
| Latency (p50, p95) | Function timing |
| Error rate | HTTP 5xx count |
| Timeouts | AbortError count |
| Availability | Health checks |

### 2. Data Quality

| Metric | Source |
|--------|--------|
| Schema failures | API validation errors |
| Missing values | Silver dq_flags |
| Invalid values | Rejected records count |
| Out-of-range inputs | Prediction warnings |
| Unknown categories | dq_flags |

### 3. Feature Monitoring

Compare recent prediction inputs vs training distribution:

- Mean, median, standard deviation
- Percentage outside training p01–p99
- Drift score (extensible via `DriftCalculator` protocol)

Future: PSI, KS distance, Jensen-Shannon divergence.

### 4. Prediction Monitoring

- Prediction distribution
- Average predicted value
- Extreme prediction rate
- Fallback prediction percentage

### 5. Retrospective Model Performance

When actual sale prices are available:

- MAE (primary), RMSE, bias, MAPE
- Metrics by region, property type, price category
- Rolling windows (7d, 30d, all_time)
- Stored in `gold.model_evaluations`

### 6. Business Monitoring (Placeholders)

Interfaces for future KPIs:
- Prediction usage rate
- Acceptance of recommended prices
- Time to sale
- Asking vs sale price gap

## Reliability Gate

Metrics based on actual prices are only reliable when:

```
sample_size >= MIN_EVALUATION_SAMPLE_SIZE (default: 30)
```

The UI shows sample size next to every metric and displays warnings when below threshold.

## Evaluation Workflow

Databricks job `evaluate_model` (scheduled daily):
1. Join `gold.predictions` + `gold.actual_sales`
2. Compute overall and segmented metrics
3. Append to `gold.model_evaluations`

## Serving Metrics Workflow

Databricks job `serving_metrics` (scheduled daily at 06:45 Europe/Amsterdam):
1. Read `gold.predictions` (successful API calls with latency)
2. Read `gold.serving_events` (failed `/api/predict` calls logged by Netlify)
3. Compute per-day request count, error/timeout counts, p50/p95 latency
4. Upsert into `gold.serving_metrics`

Failed predict requests are written to `gold.serving_events` from the Netlify function (HTTP status, error code, latency, timeout flag).

## Feature Monitoring Workflow

Databricks job `feature_monitoring` (scheduled daily):
1. Extract recent prediction payloads
2. Compare to training reference distribution
3. Write to `gold.feature_monitoring`

## Dashboard

The monitoring page displays:
- Databricks serving & API performance (latency percentiles, CPU/memory, request volume)
- Summary cards (predictions, labelled count, MAE, RMSE, bias)
- MAE by region and property type (bar charts)
- Active model version
- Warnings for low sample sizes

Serving latency is sourced from:
- Databricks `/api/2.0/serving-endpoints/{name}/metrics` (p50/p99 inside the endpoint)
- `gold.predictions.serving_latency_ms` (end-to-end API path, last 500 rows)
- Optional daily rollups in `gold.serving_metrics` when populated by jobs
