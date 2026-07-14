"""Business KPI monitoring interfaces (placeholders for production)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass
class BusinessKPIs:
    prediction_usage_count: int
    price_acceptance_rate: float | None
    avg_time_to_sale_days: float | None
    avg_asking_vs_sale_gap_pct: float | None


class BusinessMonitor(Protocol):
    def compute_kpis(self, start_date: str, end_date: str) -> BusinessKPIs: ...


class PlaceholderBusinessMonitor:
    """Demo placeholder — implement with CRM/sales data in production."""

    def compute_kpis(self, start_date: str, end_date: str) -> BusinessKPIs:
        return BusinessKPIs(
            prediction_usage_count=0,
            price_acceptance_rate=None,
            avg_time_to_sale_days=None,
            avg_asking_vs_sale_gap_pct=None,
        )
