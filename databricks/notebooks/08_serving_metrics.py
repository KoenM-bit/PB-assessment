# Databricks notebook source
# Daily rollup into gold.serving_metrics from predictions + API error events.

# COMMAND ----------

import pandas as pd

from house_price_ml.monitoring.serving import build_daily_serving_metrics

catalog = dbutils.widgets.get("catalog") or "house_price_staging"
timeout_ms = int(dbutils.widgets.get("timeout_ms") or "30000")
lookback_days = int(dbutils.widgets.get("lookback_days") or "90")

predictions = spark.table(f"{catalog}.gold.predictions").toPandas()

try:
    events = spark.table(f"{catalog}.gold.serving_events").toPandas()
except Exception:
    events = None

metrics_df = build_daily_serving_metrics(
    predictions,
    events,
    timeout_ms=timeout_ms,
    lookback_days=lookback_days,
)

if metrics_df.empty:
    dbutils.notebook.exit("No serving data in lookback window")

metrics_df["date"] = pd.to_datetime(metrics_df["date"]).dt.date
spark_metrics = spark.createDataFrame(metrics_df)
spark_metrics.createOrReplaceTempView("serving_metrics_upsert")

dates_sql = ", ".join(f"DATE '{d}'" for d in metrics_df["date"].astype(str).tolist())
spark.sql(f"DELETE FROM {catalog}.gold.serving_metrics WHERE date IN ({dates_sql})")

spark_metrics.write.format("delta").mode("append").saveAsTable(f"{catalog}.gold.serving_metrics")
