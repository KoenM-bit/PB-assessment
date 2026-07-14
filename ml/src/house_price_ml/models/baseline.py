"""Business baseline model."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


class BusinessBaseline:
    """Median price per sqm by region and property type × surface area."""

    def __init__(self) -> None:
        self.lookup: dict[str, float] = {}
        self.global_median_psm: float = 3000.0

    def fit(self, df: pd.DataFrame) -> "BusinessBaseline":
        valid = df[(df["sale_price"].notna()) & (df["surface_area"] > 0)].copy()
        valid["psm"] = valid["sale_price"] / valid["surface_area"]
        self.global_median_psm = float(valid["psm"].median()) if len(valid) else 3000.0
        grouped = valid.groupby(["region", "property_type"])["psm"].median()
        self.lookup = {f"{r}|{p}": float(v) for (r, p), v in grouped.items()}
        return self

    def _get_psm(self, region: str, property_type: str) -> float:
        return self.lookup.get(f"{region}|{property_type}", self.global_median_psm)

    def predict_row(self, row: dict[str, Any]) -> float:
        psm = self._get_psm(str(row["region"]), str(row["property_type"]))
        return float(psm * float(row["surface_area"]))

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return np.array([self.predict_row(r) for r in df.to_dict("records")])

    def save(self, path: Path) -> None:
        path.write_text(
            json.dumps({"lookup": self.lookup, "global_median_psm": self.global_median_psm})
        )

    @classmethod
    def load(cls, path: str | Path) -> "BusinessBaseline":
        p = Path(path)
        data = json.loads(p.read_text())
        model = cls()
        model.lookup = data["lookup"]
        model.global_median_psm = data["global_median_psm"]
        return model
