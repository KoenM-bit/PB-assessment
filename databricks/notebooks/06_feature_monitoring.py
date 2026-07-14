# Databricks notebook: Feature monitoring

# MAGIC %pip install /Workspace/Shared/house_price_ml-0.1.0-py3-none-any.whl

# COMMAND ----------

import json
from datetime import date, timedelta

import pandas as pd
from house_price_ml.features.pipeline import raw_to_feature_frame
from house_price_ml.monitoring.drift import compute_feature_monitoring

catalog = dbutils.widgets.get("catalog") or "house_price_staging"

features_table = spark.table(f"{catalog}.gold.listing_features").toPandas()
predictions = spark.table(f"{catalog}.gold.predictions").toPandas()

recent_rows = []
for _, row in predictions.tail(100).iterrows():
    if row.get("request_payload"):
        recent_rows.append(json.loads(row["request_payload"]))

if not recent_rows:
    dbutils.notebook.exit("No recent predictions")

recent_features = raw_to_feature_frame(recent_rows)
reference_features = features_table[
    ["surface_area", "number_of_rooms", "house_age", "energy_label_score", "dist_to_city_centre_km"]
].rename(columns={"number_of_rooms": "number_of_rooms"})

# Align column names
ref = raw_to_feature_frame(features_table.head(500).to_dict("records"))
feature_cols = ["surface_area", "number_of_rooms", "house_age", "energy_label_score", "dist_to_city_centre_km"]
mon_df = compute_feature_monitoring(ref, recent_features, feature_cols)
mon_df["monitoring_date"] = date.today()

spark.createDataFrame(mon_df).write.format("delta").mode("append").saveAsTable(
    f"{catalog}.gold.feature_monitoring"
)
