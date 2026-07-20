import { useMemo, useState } from "react";
import type { LiveLabelledSale } from "../types";
import { formatCurrency, formatDate } from "../utils/format";

const REGION_ICON = "📍";

const PROPERTY_ICONS: Record<string, string> = {
  apartment: "🏢",
  terraced_house: "🏘️",
  semi_detached: "🏠",
  detached: "🏡",
  bungalow: "🛖",
};

function propertyIcon(type: string): string {
  return PROPERTY_ICONS[type] ?? "🏠";
}

type Props = {
  items: LiveLabelledSale[];
};

export function LiveLabelledSalesTable({ items }: Props) {
  const [regionFilter, setRegionFilter] = useState<string | null>(null);
  const [typeFilter, setTypeFilter] = useState<string | null>(null);

  const regions = useMemo(
    () => [...new Set(items.map((i) => i.region))].sort(),
    [items],
  );
  const types = useMemo(
    () => [...new Set(items.map((i) => i.property_type))].sort(),
    [items],
  );

  const filtered = useMemo(() => {
    return items.filter((item) => {
      if (regionFilter && item.region !== regionFilter) return false;
      if (typeFilter && item.property_type !== typeFilter) return false;
      return true;
    });
  }, [items, regionFilter, typeFilter]);

  const toggleRegion = (region: string) => {
    setRegionFilter((current) => (current === region ? null : region));
  };

  const toggleType = (type: string) => {
    setTypeFilter((current) => (current === type ? null : type));
  };

  if (items.length === 0) return null;

  return (
    <div className="live-sales-section">
      <h3>Labelled sales detail</h3>
      <p className="muted">
        Predictions with recorded actual sale prices — filter by region or property type.
      </p>

      <div className="filter-chip-groups">
        <div className="filter-chip-group">
          <span className="filter-chip-label">Region</span>
          <button
            type="button"
            className={`filter-chip ${regionFilter === null ? "active" : ""}`}
            onClick={() => setRegionFilter(null)}
          >
            <span className="filter-chip-icon" aria-hidden>
              🌐
            </span>
            All
          </button>
          {regions.map((region) => (
            <button
              key={region}
              type="button"
              className={`filter-chip ${regionFilter === region ? "active" : ""}`}
              onClick={() => toggleRegion(region)}
            >
              <span className="filter-chip-icon" aria-hidden>
                {REGION_ICON}
              </span>
              {region}
            </button>
          ))}
        </div>
        <div className="filter-chip-group">
          <span className="filter-chip-label">Type</span>
          <button
            type="button"
            className={`filter-chip ${typeFilter === null ? "active" : ""}`}
            onClick={() => setTypeFilter(null)}
          >
            <span className="filter-chip-icon" aria-hidden>
              🌐
            </span>
            All
          </button>
          {types.map((type) => (
            <button
              key={type}
              type="button"
              className={`filter-chip ${typeFilter === type ? "active" : ""}`}
              onClick={() => toggleType(type)}
            >
              <span className="filter-chip-icon" aria-hidden>
                {propertyIcon(type)}
              </span>
              {type.replace(/_/g, " ")}
            </button>
          ))}
        </div>
      </div>

      <p className="muted filter-result-count">
        Showing {filtered.length} of {items.length} sales
      </p>

      <div className="table-wrap">
        <table className="live-sales-table">
          <thead>
            <tr>
              <th>Address</th>
              <th>Region</th>
              <th>Type</th>
              <th>m²</th>
              <th>Predicted</th>
              <th>Baseline</th>
              <th>Actual</th>
              <th>Model |err|</th>
              <th>Baseline |err|</th>
              <th>% err</th>
              <th>vs baseline</th>
              <th>Pred. date</th>
              <th>Sale date</th>
            </tr>
          </thead>
          <tbody>
            {filtered.map((row) => (
              <tr key={row.prediction_id}>
                <td>{row.address}</td>
                <td>
                  <span className="cell-with-icon">
                    <span aria-hidden>{REGION_ICON}</span>
                    {row.region}
                  </span>
                </td>
                <td>
                  <span className="cell-with-icon">
                    <span aria-hidden>{propertyIcon(row.property_type)}</span>
                    {row.property_type.replace(/_/g, " ")}
                  </span>
                </td>
                <td>{row.surface_area}</td>
                <td>{formatCurrency(row.predicted_price)}</td>
                <td>{formatCurrency(row.baseline_price)}</td>
                <td>{formatCurrency(row.actual_sale_price)}</td>
                <td>{formatCurrency(row.model_abs_error)}</td>
                <td>{formatCurrency(row.baseline_abs_error)}</td>
                <td>{row.model_pct_error.toFixed(1)}%</td>
                <td>
                  {row.beats_baseline ? (
                    <span className="trend-badge up">✓ model</span>
                  ) : (
                    <span className="trend-badge down">baseline</span>
                  )}
                </td>
                <td>{formatDate(row.prediction_date)}</td>
                <td>{row.sale_date ? formatDate(row.sale_date) : "—"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
