import { useMemo, useState } from "react";
import type { LiveLabelledSale } from "../types";
import { formatCurrency, formatDate, formatEuroK, improvementEur } from "../utils/format";
import { computeLiveSalesHeadToHead } from "../utils/liveSalesHeadToHead";

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

  const headToHead = useMemo(() => computeLiveSalesHeadToHead(filtered), [filtered]);

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
        {headToHead.ties > 0 ? ` (${headToHead.ties} tie${headToHead.ties === 1 ? "" : "s"} excluded from win %)` : ""}
      </p>

      <div className="live-head-to-head metrics-grid" aria-live="polite">
        <div className="metric-card live-h2h-card live-h2h-model">
          <div className="metric-value">{headToHead.sample_size > 0 ? `${headToHead.model_better_pct}%` : "—"}</div>
          <div className="metric-label">Model beter</div>
          {headToHead.sample_size > 0 && (
            <div className="metric-label muted-small">{headToHead.model_wins} van {headToHead.sample_size - headToHead.ties}</div>
          )}
        </div>
        <div className="metric-card live-h2h-card live-h2h-baseline">
          <div className="metric-value">{headToHead.sample_size > 0 ? `${headToHead.baseline_better_pct}%` : "—"}</div>
          <div className="metric-label">Baseline beter</div>
          {headToHead.sample_size > 0 && (
            <div className="metric-label muted-small">{headToHead.baseline_wins} van {headToHead.sample_size - headToHead.ties}</div>
          )}
        </div>
        <div className="metric-card live-h2h-card">
          <div className="metric-value trend-badge up">
            {headToHead.avg_win_when_model_better_eur != null
              ? formatCurrency(headToHead.avg_win_when_model_better_eur)
              : "—"}
          </div>
          <div className="metric-label">Gem. winst als model wint</div>
          <div className="metric-label muted-small">lagere |fout| t.o.v. baseline</div>
        </div>
        <div className="metric-card live-h2h-card">
          <div className="metric-value trend-badge down">
            {headToHead.avg_loss_when_baseline_better_eur != null
              ? formatCurrency(headToHead.avg_loss_when_baseline_better_eur)
              : "—"}
          </div>
          <div className="metric-label">Gem. verlies als baseline wint</div>
          <div className="metric-label muted-small">extra |fout| t.o.v. baseline</div>
        </div>
      </div>

      <div className="table-wrap">
        <table className="live-sales-table">
          <caption className="live-sales-caption">
            Improvement = Baseline error − Model error (positief = model heeft lagere absolute fout)
          </caption>
          <thead>
            <tr>
              <th>Address</th>
              <th>Region</th>
              <th>Type</th>
              <th>m²</th>
              <th>Predicted</th>
              <th className="live-sales-metric-col">Actual</th>
              <th className="live-sales-metric-col">Model err</th>
              <th className="live-sales-metric-col">Baseline err</th>
              <th className="live-sales-metric-col">Δ Improvement</th>
              <th>Pred. date</th>
              <th>Sale date</th>
            </tr>
          </thead>
          <tbody>
            {filtered.map((row) => {
              const delta = improvementEur(row.baseline_abs_error, row.model_abs_error);
              const deltaClass =
                delta > 0 ? "delta-improvement positive" : delta < 0 ? "delta-improvement negative" : "delta-improvement";
              return (
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
                <td className="live-sales-metric-col">{formatEuroK(row.actual_sale_price)}</td>
                <td className="live-sales-metric-col">{formatEuroK(row.model_abs_error)}</td>
                <td className="live-sales-metric-col">{formatEuroK(row.baseline_abs_error)}</td>
                <td className={`live-sales-metric-col ${deltaClass}`} title={formatCurrency(delta)}>
                  {formatEuroK(delta, true)}
                </td>
                <td>{formatDate(row.prediction_date)}</td>
                <td>{row.sale_date ? formatDate(row.sale_date) : "—"}</td>
              </tr>
            );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
