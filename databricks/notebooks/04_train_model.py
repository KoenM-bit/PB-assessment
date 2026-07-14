# Databricks notebook source
# Train and register model in Unity Catalog

# MAGIC %pip install /Workspace/Shared/house_price_ml-0.1.0-py3-none-any.whl

# COMMAND ----------

from pathlib import Path

import mlflow
from mlflow import MlflowClient
from house_price_ml.models.train import train

catalog = dbutils.widgets.get("catalog") or "house_price_staging"
model_alias = dbutils.widgets.get("model_alias") or "challenger"

# Export gold features to temp CSV for training
gold_df = spark.table(f"{catalog}.gold.listing_features").toPandas()
silver_df = spark.table(f"{catalog}.silver.listings_clean").toPandas()
merged = silver_df.merge(gold_df, on="listing_id", suffixes=("", "_gold"))
tmp_path = "/tmp/train_data.csv"
merged.to_csv(tmp_path, index=False)

out = Path("/tmp/model_output")
train(Path(tmp_path), "random_forest", out)

model_name = f"{catalog}.gold.house_price_model"
model_uri = f"file://{out / 'mlflow_model'}"

mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri("databricks-uc")

print(f"Registering {model_name} from {model_uri}")
registered = mlflow.register_model(model_uri=model_uri, name=model_name)
version = str(registered.version)
print(f"Registered version {version}")

client = MlflowClient(registry_uri="databricks-uc")
client.set_registered_model_alias(model_name, model_alias, version)
print(f"Alias {model_alias} -> version {version}")
