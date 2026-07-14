# Databricks notebook source
# Gold features — wheel via serverless environment dependencies (databricks.yml).

# COMMAND ----------

from house_price_ml.features.aggregates import silver_to_gold_features

catalog = dbutils.widgets.get("catalog") or "house_price_staging"
silver_df = spark.table(f"{catalog}.silver.listings_clean").toPandas()
gold_df = silver_to_gold_features(silver_df)
spark.createDataFrame(gold_df).write.format("delta").mode("overwrite").option(
    "overwriteSchema", "true"
).saveAsTable(f"{catalog}.gold.listing_features")
