"""Application configuration.

Import settings explicitly from house_price_ml.config.settings when needed.
Avoid eager imports here so Model Serving bundles do not pull training-only deps.
"""

from .constants import PRICE_CATEGORY_BOUNDS as PRICE_CATEGORY_BOUNDS
