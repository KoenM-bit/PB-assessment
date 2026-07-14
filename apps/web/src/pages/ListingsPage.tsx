import { useCallback, useEffect, useState } from "react";
import { api } from "../api/client";
import type { PredictionListItem } from "../types";
import { formatCurrency, formatDate, formatPercent } from "../utils/format";

const WRITE_TOKEN_KEY = "demo_write_token";

function loadWriteToken(): string {
  try {
    return sessionStorage.getItem(WRITE_TOKEN_KEY) ?? "";
  } catch {
    return "";
  }
}

function saveWriteToken(token: string): void {
  try {
    sessionStorage.setItem(WRITE_TOKEN_KEY, token);
  } catch {
    // ignore storage errors in private browsing
  }
}

interface SaleRowEditorProps {
  item: PredictionListItem;
  writeToken: string;
  onSaved: () => Promise<void>;
}

function SaleRowEditor({ item, writeToken, onSaved }: SaleRowEditorProps) {
  const [price, setPrice] = useState(item.predicted_price);
  const [saleDate, setSaleDate] = useState(new Date().toISOString().slice(0, 10));
  const [saving, setSaving] = useState(false);
  const [rowError, setRowError] = useState<string | null>(null);

  const handleSave = async () => {
    if (!writeToken.trim()) {
      setRowError("Enter the demo write token above first.");
      return;
    }
    setSaving(true);
    setRowError(null);
    try {
      await api.submitActualSale(
        {
          prediction_id: item.prediction_id,
          actual_sale_price: price,
          sale_date: saleDate,
        },
        writeToken.trim(),
      );
      await onSaved();
    } catch (err) {
      setRowError(err instanceof Error ? err.message : "Failed to record sale");
    } finally {
      setSaving(false);
    }
  };

  return (
    <>
      <td>
        <input
          type="number"
          className="table-input"
          value={price}
          min={1}
          onChange={(e) => setPrice(Number(e.target.value))}
          aria-label={`Actual sale price for ${item.address}`}
        />
      </td>
      <td>
        <input
          type="date"
          className="table-input"
          value={saleDate}
          onChange={(e) => setSaleDate(e.target.value)}
          aria-label={`Sale date for ${item.address}`}
        />
      </td>
      <td>—</td>
      <td>—</td>
      <td>
        <button
          type="button"
          className="btn-sm"
          disabled={saving || !writeToken.trim()}
          onClick={handleSave}
        >
          {saving ? "Saving…" : "Record sale"}
        </button>
        {rowError && <div className="row-error">{rowError}</div>}
      </td>
    </>
  );
}

export function ListingsPage() {
  const [items, setItems] = useState<PredictionListItem[]>([]);
  const [writeToken, setWriteToken] = useState(loadWriteToken);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [successMessage, setSuccessMessage] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    const data = await api.getPredictions();
    setItems(data.items);
  }, []);

  useEffect(() => {
    refresh()
      .catch((err) => setError(err instanceof Error ? err.message : "Failed to load predictions"))
      .finally(() => setLoading(false));
  }, [refresh]);

  useEffect(() => {
    saveWriteToken(writeToken);
  }, [writeToken]);

  const handleSaved = async () => {
    setSuccessMessage("Sale recorded — monitoring metrics will update after evaluation runs.");
    await refresh();
    setTimeout(() => setSuccessMessage(null), 4000);
  };

  if (loading) return <div className="card">Loading predictions…</div>;
  if (error) return <div className="error">{error}</div>;

  const labelledCount = items.filter((item) => item.actual_sale_price != null).length;

  return (
    <div className="card listings-card">
      <h2>Predictions &amp; Sales</h2>
      <p className="muted">
        Record actual sale prices directly on each prediction. {labelledCount} of {items.length}{" "}
        predictions have a recorded sale.
      </p>

      <div className="listings-toolbar">
        <label>
          Demo write token
          <input
            type="password"
            value={writeToken}
            onChange={(e) => setWriteToken(e.target.value)}
            placeholder="DEMO_WRITE_TOKEN"
          />
        </label>
      </div>

      {successMessage && <div className="success">{successMessage}</div>}

      {items.length === 0 ? (
        <p>No predictions yet. Submit a new prediction first.</p>
      ) : (
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Address</th>
                <th>Region</th>
                <th>Type</th>
                <th>Area</th>
                <th>Predicted</th>
                <th>Actual</th>
                <th>Sale date</th>
                <th>Abs error</th>
                <th>% error</th>
                <th>Action</th>
              </tr>
            </thead>
            <tbody>
              {items.map((item) => (
                <tr key={item.prediction_id}>
                  <td>
                    <div className="address-cell">{item.address}</div>
                    {item.postcode && <small>{item.postcode}</small>}
                    <div className="muted">
                      <code>{item.prediction_id.slice(0, 8)}…</code>
                      <span className="badge">{item.model_version}</span>
                    </div>
                    <small className="muted">Predicted {formatDate(item.prediction_date)}</small>
                  </td>
                  <td>{item.region}</td>
                  <td>{item.property_type.replace(/_/g, " ")}</td>
                  <td>{item.surface_area} m²</td>
                  <td>{formatCurrency(item.predicted_price)}</td>
                  {item.actual_sale_price != null ? (
                    <>
                      <td>{formatCurrency(item.actual_sale_price)}</td>
                      <td>{item.sale_date ? formatDate(item.sale_date) : "—"}</td>
                      <td>{item.absolute_error != null ? formatCurrency(item.absolute_error) : "—"}</td>
                      <td>{formatPercent(item.percentage_error)}</td>
                      <td><span className="badge badge-success">Recorded</span></td>
                    </>
                  ) : (
                    <SaleRowEditor
                      item={item}
                      writeToken={writeToken}
                      onSaved={handleSaved}
                    />
                  )}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
