# Databricks notebook source
# Silver clean

# MAGIC %run ./_install_wheel

# COMMAND ----------

import pandas as pd
from house_price_ml.data.silver import bronze_to_silver

catalog = dbutils.widgets.get("catalog") or "house_price_staging"
bronze_df = spark.table(f"{catalog}.bronze.listings_raw").toPandas()
clean, rejected = bronze_to_silver(bronze_df)

spark.createDataFrame(clean).write.format("delta").mode("overwrite").saveAsTable(
    f"{catalog}.silver.listings_clean"
)
if len(rejected) > 0:
    spark.createDataFrame(rejected).write.format("delta").mode("append").saveAsTable(
        f"{catalog}.silver.listings_rejected"
    )
