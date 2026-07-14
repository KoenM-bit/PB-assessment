"""Energy label feature mappings."""

from __future__ import annotations

from house_price_ml.config.constants import ENERGY_LABEL_SCORES


def energy_label_to_score(label: str) -> int:
    """Convert energy label to numeric score."""
    return ENERGY_LABEL_SCORES.get(label, 0)
