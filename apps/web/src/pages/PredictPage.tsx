import { useState } from "react";
import { api } from "../api/client";
import { PredictionResultCard } from "../components/PredictionResultCard";
import type { PredictRequest, PredictionResult } from "../types";

const REGIONS = ["Amsterdam", "Rotterdam", "Utrecht", "The Hague", "Eindhoven", "Groningen", "Maastricht", "Nijmegen"];
const PROPERTY_TYPES = ["apartment", "terraced_house", "semi_detached", "detached", "bungalow"];
const ENERGY_LABELS = ["A++", "A+", "A", "B", "C", "D", "E", "F", "G"];

const DEFAULT_FORM: PredictRequest = {
  address: "Domstraat 12",
  postcode: "3512 JC",
  surface_area: 120,
  number_of_rooms: 5,
  number_of_bedrooms: 3,
  build_year: 1985,
  energy_label: "B",
  property_type: "terraced_house",
  garden: true,
  region: "Utrecht",
  latitude: 52.0907,
  longitude: 5.1214,
};

export function PredictPage() {
  const [form, setForm] = useState<PredictRequest>(DEFAULT_FORM);
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const update = (field: keyof PredictRequest, value: string | number | boolean) => {
    setForm((prev) => ({ ...prev, [field]: value }));
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const data = await api.predict({
        ...form,
        postcode: form.postcode?.trim() || undefined,
        prediction_date: new Date().toISOString().slice(0, 10),
      });
      setResult(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Prediction failed");
    } finally {
      setLoading(false);
    }
  };

  return (
    <>
      <div className="card">
        <h2>New Prediction</h2>
        <form onSubmit={handleSubmit}>
          <div className="form-grid">
            <label className="full-width">
              Address
              <input type="text" value={form.address}
                onChange={(e) => update("address", e.target.value)}
                placeholder="Street and house number" required />
            </label>
            <label>
              Postcode
              <input type="text" value={form.postcode ?? ""}
                onChange={(e) => update("postcode", e.target.value)}
                placeholder="3512 JC" />
            </label>
            <label>
              Surface area (m²)
              <input type="number" value={form.surface_area} min={1}
                onChange={(e) => update("surface_area", Number(e.target.value))} required />
            </label>
            <label>
              Rooms
              <input type="number" value={form.number_of_rooms} min={1}
                onChange={(e) => update("number_of_rooms", Number(e.target.value))} required />
            </label>
            <label>
              Bedrooms
              <input type="number" value={form.number_of_bedrooms} min={0}
                onChange={(e) => update("number_of_bedrooms", Number(e.target.value))} required />
            </label>
            <label>
              Build year
              <input type="number" value={form.build_year}
                onChange={(e) => update("build_year", Number(e.target.value))} required />
            </label>
            <label>
              Energy label
              <select value={form.energy_label} onChange={(e) => update("energy_label", e.target.value)}>
                {ENERGY_LABELS.map((l) => <option key={l} value={l}>{l}</option>)}
              </select>
            </label>
            <label>
              Property type
              <select value={form.property_type} onChange={(e) => update("property_type", e.target.value)}>
                {PROPERTY_TYPES.map((t) => <option key={t} value={t}>{t.replace(/_/g, " ")}</option>)}
              </select>
            </label>
            <label>
              Region
              <select value={form.region} onChange={(e) => update("region", e.target.value)}>
                {REGIONS.map((r) => <option key={r} value={r}>{r}</option>)}
              </select>
            </label>
            <label>
              Latitude
              <input type="number" step="0.0001" value={form.latitude}
                onChange={(e) => update("latitude", Number(e.target.value))} required />
            </label>
            <label>
              Longitude
              <input type="number" step="0.0001" value={form.longitude}
                onChange={(e) => update("longitude", Number(e.target.value))} required />
            </label>
            <label>
              Garden
              <select value={form.garden ? "yes" : "no"}
                onChange={(e) => update("garden", e.target.value === "yes")}>
                <option value="yes">Yes</option>
                <option value="no">No</option>
              </select>
            </label>
          </div>
          <button type="submit" disabled={loading}>
            {loading ? "Predicting…" : "Get Prediction"}
          </button>
        </form>
        {error && <div className="error">{error}</div>}
      </div>

      {result && (
        <PredictionResultCard
          predictedPrice={result.predicted_price}
          modelVersion={result.model_version}
          modelAlias={result.model_alias}
          timestamp={result.prediction_timestamp}
          predictionId={result.prediction_id}
          address={result.address}
          propertyKey={result.property_key}
          warnings={result.warnings}
        />
      )}
    </>
  );
}
