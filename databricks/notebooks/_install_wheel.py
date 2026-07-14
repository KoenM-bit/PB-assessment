# Databricks notebook source
"""Install house_price_ml wheel from workspace path (wheel_path widget)."""
import subprocess
import sys

wheel_path = ""
try:
    wheel_path = dbutils.widgets.get("wheel_path")
except Exception:
    pass
if not wheel_path:
    user = dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()
    wheel_path = f"/Workspace/Users/{user}/libs/house_price_ml-0.1.0-py3-none-any.whl"

print(f"Installing wheel: {wheel_path}")
subprocess.check_call([sys.executable, "-m", "pip", "install", wheel_path, "-q"])
print("Wheel installed.")
