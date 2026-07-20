import type { Handler, HandlerEvent } from "@netlify/functions";
import { v4 as uuidv4 } from "uuid";
import { getConfig } from "./_shared/config.js";
import { getActualSales, getFeatureMonitoring, getPredictions, getServingMetricsHistory, isBaselineFallback, isPeerServingFallback } from "./_shared/databricks.js";
import { handleError, successResponse } from "./_shared/errors.js";
import {
  baselinePredict,
  loadTrainingManifest,
  type MetricSet,
  type TrainingManifest,
} from "./_shared/training_manifest.js";
import {
  dailyServingFromPredictions,
  fetchDatabricksEndpointMetrics,
  recentRequestLatencies,
  summarizeLatencies,
} from "./_shared/serving_performance.js";
import { buildRequestMonitoring } from "./_shared/request_monitoring.js";
import { buildLiveLabelledMonitoring } from "./_shared/live_evaluation_monitoring.js";

function computeMetrics(
  items: { predicted: number; actual: number }[],
): { mae: number; rmse: number; bias: number; mape: number; sample_size: number } {
  if (items.length === 0) {
    return { mae: 0, rmse: 0, bias: 0, mape: 0, sample_size: 0 };
  }
  const errors = items.map((i) => i.actual - i.predicted);
  const absErrors = errors.map(Math.abs);
  const mae = absErrors.reduce((a, b) => a + b, 0) / items.length;
  const rmse = Math.sqrt(errors.reduce((a, b) => a + b * b, 0) / items.length);
  const bias = errors.reduce((a, b) => a + b, 0) / items.length;
  const mape =
    items.reduce((sum, i) => sum + Math.abs((i.actual - i.predicted) / i.actual), 0) /
    items.length *
    100;
  return { mae, rmse, bias, mape, sample_size: items.length };
}

function compareEvaluation(
  model: MetricSet,
  baseline: MetricSet,
): { beats_baseline: boolean; mae_improvement_pct: number } {
  const beats = model.mae < baseline.mae;
  const improvement = baseline.mae > 0 ? (1 - model.mae / baseline.mae) * 100 : 0;
  return { beats_baseline: beats, mae_improvement_pct: Math.round(improvement * 10) / 10 };
}

function buildLiveComparison(
  labelled: { predicted: number; actual: number; baseline: number }[],
  minSample: number,
) {
  const modelMetrics = computeMetrics(
    labelled.map((item) => ({ predicted: item.predicted, actual: item.actual })),
  );
  const baselineMetrics = computeMetrics(
    labelled.map((item) => ({ predicted: item.baseline, actual: item.actual })),
  );
  const comparison = compareEvaluation(modelMetrics, baselineMetrics);
  return {
    model: modelMetrics,
    baseline: baselineMetrics,
    ...comparison,
    is_reliable: modelMetrics.sample_size >= minSample,
  };
}

function featureBoundsSummary(manifest: TrainingManifest) {
  const priority = [
    "surface_area",
    "house_age",
    "dist_to_city_centre_km",
    "region_median_price_per_sqm",
    "energy_label_score",
    "number_of_rooms",
  ];
  return priority
    .filter((name) => manifest.feature_bounds[name])
    .map((name) => ({
      feature: name,
      p01: manifest.feature_bounds[name].p01,
      p99: manifest.feature_bounds[name].p99,
    }));
}

export const handler: Handler = async (event: HandlerEvent) => {
  const requestId = uuidv4();
  try {
    const config = getConfig();
    const manifest = loadTrainingManifest();
    const [predictions, actuals, endpointMetrics, servingMetricsHistory, featureMonitoring] =
      await Promise.all([
      getPredictions(config, 500),
      getActualSales(config),
      fetchDatabricksEndpointMetrics(config),
      getServingMetricsHistory(config, 30),
      getFeatureMonitoring(config, 30),
    ]);
    const actualMap = new Map(actuals.map((a) => [a.prediction_id, a]));

    const labelled = predictions
      .filter((p) => actualMap.has(p.prediction_id))
      .map((p) => {
        const actual = actualMap.get(p.prediction_id)!;
        return {
          predicted: p.predicted_price,
          actual: actual.actual_sale_price,
          baseline: baselinePredict(p.region, p.property_type, p.surface_area, manifest),
          region: p.region,
          property_type: p.property_type,
          prediction_id: p.prediction_id,
          address: p.address,
          surface_area: p.surface_area,
          prediction_timestamp: p.prediction_timestamp,
          sale_date: actual.sale_date ?? null,
        };
      });

    const liveLabelledMonitoring = buildLiveLabelledMonitoring(labelled);

    const liveOverall = buildLiveComparison(labelled, config.minEvaluationSampleSize);

    const liveByRegion: Record<string, ReturnType<typeof buildLiveComparison>> = {};
    for (const region of [...new Set(labelled.map((l) => l.region))]) {
      liveByRegion[region] = buildLiveComparison(
        labelled.filter((l) => l.region === region),
        config.minEvaluationSampleSize,
      );
    }

    const overall = computeMetrics(
      labelled.map((l) => ({ predicted: l.predicted, actual: l.actual })),
    );
    const minSample = config.minEvaluationSampleSize;
    const isReliable = overall.sample_size >= minSample;

    const byRegion: Record<string, ReturnType<typeof computeMetrics>> = {};
    for (const region of [...new Set(labelled.map((l) => l.region))]) {
      const subset = labelled.filter((l) => l.region === region);
      byRegion[region] = computeMetrics(
        subset.map((l) => ({ predicted: l.predicted, actual: l.actual })),
      );
    }

    const byPropertyType: Record<string, ReturnType<typeof computeMetrics>> = {};
    for (const pt of [...new Set(labelled.map((l) => l.property_type))]) {
      const subset = labelled.filter((l) => l.property_type === pt);
      byPropertyType[pt] = computeMetrics(
        subset.map((l) => ({ predicted: l.predicted, actual: l.actual })),
      );
    }

    const activeModel = predictions[0]?.model_version ?? "unknown";
    const warnings: string[] = [];
    if (!isReliable) {
      warnings.push(
        `Live sample size ${overall.sample_size} below minimum ${minSample} — production metrics not conclusive`,
      );
    }

    const predictionLatencies = predictions.map((p) => p.serving_latency_ms).filter((ms) => ms > 0);
    const apiLatency = summarizeLatencies(predictionLatencies);
    const baselineFallbackCount = predictions.filter((p) => isBaselineFallback(p)).length;
    const peerFallbackCount = predictions.filter((p) => isPeerServingFallback(p)).length;
    const dailyFromPredictions = dailyServingFromPredictions(predictions);

    const endpointRequestTotal = endpointMetrics?.request_count_total ?? 0;
    const endpointErrors = (endpointMetrics?.error_4xx_total ?? 0) + (endpointMetrics?.error_5xx_total ?? 0);
    const rolledRequestCount = servingMetricsHistory.reduce((sum, row) => sum + row.request_count, 0);
    const rolledErrorCount = servingMetricsHistory.reduce((sum, row) => sum + row.error_count, 0);
    const errorRate =
      servingMetricsHistory.length > 0 && rolledRequestCount + rolledErrorCount > 0
        ? rolledErrorCount / (rolledRequestCount + rolledErrorCount)
        : endpointRequestTotal > 0
          ? endpointErrors / endpointRequestTotal
          : 0;

    const historyChart =
      servingMetricsHistory.length > 0
        ? [...servingMetricsHistory]
            .sort((a, b) => a.date.localeCompare(b.date))
            .map((row) => ({
              date: row.date,
              request_count: row.request_count,
              p50_ms: Math.round(row.p50_latency_ms),
              p95_ms: Math.round(row.p95_latency_ms),
              error_count: row.error_count,
              timeout_count: row.timeout_count,
            }))
        : dailyFromPredictions.map((row) => ({
            date: row.date,
            request_count: row.request_count,
            p50_ms: row.p50_ms,
            p95_ms: row.p95_ms,
            error_count: 0,
            timeout_count: 0,
          }));

    const requestMonitoring = buildRequestMonitoring(predictions, manifest);
    for (const w of requestMonitoring.warnings) {
      if (!warnings.includes(w)) warnings.push(w);
    }

    const latestFeatureDate = featureMonitoring[0]?.monitoring_date;
    const featureRows = latestFeatureDate
      ? featureMonitoring.filter((row) => row.monitoring_date === latestFeatureDate)
      : [];

    return successResponse(
      {
        summary: {
          total_predictions: predictions.length,
          labelled_predictions: labelled.length,
          active_model_version: activeModel,
          min_sample_size: minSample,
        },
        training: {
          model_type: manifest.model_type,
          feature_pipeline_version: manifest.feature_pipeline_version,
          training_date: manifest.training_date,
          git_commit: manifest.git_commit,
          training_data_rows: manifest.training_data_rows,
          test_rows: manifest.test_rows,
          validation_approach: manifest.validation_approach,
          regions: manifest.regions,
          property_types: manifest.property_types,
          surface_area_range: manifest.surface_area_range,
          price_range: manifest.price_range,
          feature_bounds_summary: featureBoundsSummary(manifest),
          walk_forward_baseline_mae_mean: manifest.walk_forward_baseline_mae_mean,
        },
        holdout_evaluation: manifest.holdout_evaluation,
        live_evaluation: {
          overall: liveOverall,
          by_region: liveByRegion,
        },
        performance: {
          overall: { ...overall, is_reliable: isReliable },
          by_region: byRegion,
          by_property_type: byPropertyType,
        },
        infrastructure: {
          request_count: predictions.length,
          error_rate: Math.round(errorRate * 1000) / 1000,
          timeout_rate:
            servingMetricsHistory.length > 0
              ? servingMetricsHistory.reduce((sum, row) => sum + row.timeout_count, 0) /
                Math.max(
                  servingMetricsHistory.reduce((sum, row) => sum + row.request_count, 0),
                  1,
                )
              : 0,
          api_latency: apiLatency,
          fallback_rate:
            predictions.length > 0 ? baselineFallbackCount / predictions.length : 0,
          peer_fallback_rate:
            predictions.length > 0 ? peerFallbackCount / predictions.length : 0,
          daily: dailyFromPredictions,
          history: historyChart,
          recent_requests: recentRequestLatencies(predictions, 5),
          databricks_endpoint: endpointMetrics,
          serving_endpoint: config.servingEndpoint,
        },
        data_quality: {
          missing_value_rate: 0,
          out_of_range_rate:
            predictions.filter((p) => p.warnings.length > 0).length /
            Math.max(predictions.length, 1),
        },
        prediction_distribution: {
          mean: predictions.length
            ? predictions.reduce((a, p) => a + p.predicted_price, 0) / predictions.length
            : 0,
          count: predictions.length,
        },
        feature_monitoring: featureRows,
        request_monitoring: requestMonitoring,
        live_labelled_sales: liveLabelledMonitoring,
        warnings,
      },
      requestId,
    );
  } catch (err) {
    return handleError(err, requestId);
  }
};
