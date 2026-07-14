# Databricks notebook: Train and register model

# MAGIC %pip install /Workspace/Shared/house_price_ml-0.1.0-py3-none-any.whl

# COMMAND ----------

from pathlib import Path
from house_price_ml.models.train import train

catalog = dbutils.widgets.get("catalog") or "house_price_staging"
model_alias = dbutils.widgets.get("model_alias") or "challenger"

# Export gold features to temp CSV for training
gold_df = spark.table(f"{catalog}.gold.listing_features").toPandas()
# Merge back silver fields needed for training
silver_df = spark.table(f"{catalog}.silver.listings_clean").toPandas()
merged = silver_df.merge(gold_df, on="listing_id", suffixes=("", "_gold"))
tmp_path = "/tmp/train_data.csv"
merged.to_csv(tmp_path, index=False)

out = Path("/tmp/model_output")
train(Path(tmp_path), "random_forest", out)

# Register in MLflow (Databricks tracks experiments automatically)
import mlflow
client = mlflow.tracking.MlflowClient()
# In production: client.set_registered_model_alias("house_price_model", model_alias, version)
