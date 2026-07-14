# Databricks notebook source
# Retrospective evaluation — wheel via serverless environment dependencies.

# COMMAND ----------

from house_price_ml.monitoring.builders import build_retrospective_evaluations

catalog = dbutils.widgets.get("catalog") or "house_price_staging"

predictions = spark.table(f"{catalog}.gold.predictions").toPandas()
actuals = spark.table(f"{catalog}.gold.actual_sales").toPandas()

if len(actuals) == 0:
    dbutils.notebook.exit("No actual sales to evaluate")

joined = predictions.merge(actuals, on="prediction_id", how="inner")
if "region" not in joined.columns:
    import json
    joined["region"] = joined["request_payload"].apply(
        lambda x: json.loads(x).get("region", "unknown") if x else "unknown"
    )
    joined["property_type"] = joined["request_payload"].apply(
        lambda x: json.loads(x).get("property_type", "unknown") if x else "unknown"
    )

model_version = joined["model_version"].iloc[0] if len(joined) else "unknown"
eval_df = build_retrospective_evaluations(joined, model_version)

spark.createDataFrame(eval_df).write.format("delta").mode("append").saveAsTable(
    f"{catalog}.gold.model_evaluations"
)
