"""MLflow serving wrapper."""

from __future__ import annotations

import json
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline

from house_price_ml.features.pipeline import build_preprocessor
from house_price_ml.models.baseline import BusinessBaseline

SERVING_PIP_REQUIREMENTS = [
    f"mlflow=={mlflow.__version__}",
    "pandas>=2.0",
    "numpy>=1.24",
    "scikit-learn>=1.3",
    "scipy>=1.10",
    "joblib>=1.3",
]

PYFUNC_CODE_RELPATH = "code/pyfunc_model.py"

# Only package modules required at inference time (keeps UC artifact lean).
SERVING_PACKAGE_FILES = [
    "__init__.py",
    "config/__init__.py",
    "config/constants.py",
    "features/__init__.py",
    "features/pipeline.py",
    "features/energy.py",
    "features/geo.py",
    "models/__init__.py",
    "models/baseline.py",
]


def build_sklearn_pipeline(estimator: Any | None = None) -> Pipeline:
    est = estimator if estimator is not None else RandomForestRegressor(
        n_estimators=100, random_state=42, n_jobs=-1
    )
    return Pipeline([("preprocessor", build_preprocessor()), ("estimator", est)])


def default_serving_input() -> pd.DataFrame:
    """Representative raw listing row for MLflow signature / input example."""
    return pd.DataFrame(
        [
            {
                "surface_area": 120.0,
                "number_of_rooms": 5,
                "number_of_bedrooms": 3,
                "build_year": 1985,
                "energy_label": "B",
                "property_type": "terraced_house",
                "garden": True,
                "region": "Utrecht",
                "latitude": 52.0907,
                "longitude": 5.1214,
                "prediction_date": "2026-07-14",
            }
        ]
    )


def _stage_bundle_dir(staging: Path) -> tuple[Path, Path]:
    serving_dir = Path(__file__).resolve().parent
    src_pkg = serving_dir.parents[1] / "house_price_ml"
    bundle = staging / "bundle"
    bundle.mkdir(parents=True)
    dest_pkg = bundle / "house_price_ml"
    dest_pkg.mkdir(parents=True)
    for rel in SERVING_PACKAGE_FILES:
        src = src_pkg / rel
        dst = dest_pkg / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    shutil.copy2(serving_dir / "pyfunc_model.py", bundle / "pyfunc_model.py")
    return bundle / "pyfunc_model.py", dest_pkg


def _validate_serving_import(model_dir: Path) -> None:
    """Simulate Databricks import of code/pyfunc_model.py before UC registration."""
    import subprocess
    import sys

    code_dir = model_dir / "code"
    script = (
        "import sys\n"
        f"sys.path.insert(0, {str(code_dir)!r})\n"
        "import pyfunc_model  # noqa: F401\n"
        "print('serving import ok')\n"
    )
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            "Serving bundle import validation failed:\n"
            f"{result.stderr or result.stdout}"
        )


def _normalize_model_code_path(model_dir: Path) -> None:
    """Databricks needs a relative model_code_path inside the registered artifact."""
    code_dir = model_dir / "code"
    target = code_dir / "pyfunc_model.py"
    if not target.exists():
        raise RuntimeError(f"Missing {PYFUNC_CODE_RELPATH} in model artifact")

    mlmodel = model_dir / "MLmodel"
    text = mlmodel.read_text()
    text = re.sub(
        r"^(\s*)model_code_path:.*$",
        rf"\1model_code_path: {PYFUNC_CODE_RELPATH}",
        text,
        flags=re.MULTILINE,
    )
    mlmodel.write_text(text)


def save_model_artifact(
    pipeline: Pipeline,
    baseline: BusinessBaseline,
    metadata: dict[str, Any],
    output_dir: str,
) -> None:
    """Log complete serving model to MLflow."""
    import mlflow.sklearn as mlflow_sklearn
    from mlflow.models import infer_signature

    out = Path(output_dir)
    if out.exists():
        shutil.rmtree(out)

    with tempfile.TemporaryDirectory() as staging, tempfile.TemporaryDirectory() as tmp:
        staging_path = Path(staging)
        tmp_path = Path(tmp)
        pyfunc_entrypoint, package_path = _stage_bundle_dir(staging_path)

        sklearn_path = tmp_path / "sklearn_model"
        baseline_path = tmp_path / "baseline.json"
        metadata_path = tmp_path / "metadata.json"

        mlflow_sklearn.save_model(
            pipeline,
            str(sklearn_path),
            serialization_format=mlflow_sklearn.SERIALIZATION_FORMAT_PICKLE,
            pip_requirements=SERVING_PIP_REQUIREMENTS,
        )
        baseline.save(baseline_path)
        metadata_path.write_text(json.dumps(metadata))

        sample_input = default_serving_input()
        sample_output = pd.DataFrame(
            {"predicted_price": [350000.0], "warnings": ['["example"]']}
        )
        signature = infer_signature(sample_input, sample_output)

        mlflow.pyfunc.save_model(
            path=str(out),
            python_model=str(pyfunc_entrypoint),
            artifacts={
                "sklearn_model": str(sklearn_path),
                "baseline": str(baseline_path),
                "metadata": str(metadata_path),
            },
            signature=signature,
            input_example=sample_input,
            code_paths=[str(package_path)],
            pip_requirements=SERVING_PIP_REQUIREMENTS,
        )

        code_dir = out / "code"
        code_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(pyfunc_entrypoint, code_dir / "pyfunc_model.py")

    _normalize_model_code_path(out)
    _validate_serving_import(out)

    if not (out / "code" / "house_price_ml").exists():
        raise RuntimeError("Missing code/house_price_ml in model artifact")

    for line in (out / "MLmodel").read_text().splitlines():
        if "model_code_path:" in line:
            path_value = line.split(":", 1)[1].strip()
            if path_value.startswith("/"):
                raise RuntimeError(f"Absolute model_code_path remains: {path_value}")
            if path_value != PYFUNC_CODE_RELPATH:
                raise RuntimeError(f"Unexpected model_code_path: {path_value}")
