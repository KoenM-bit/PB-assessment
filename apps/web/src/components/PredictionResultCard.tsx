import { formatCurrency, formatDate } from "../utils/format";

interface PredictionResultCardProps {
  predictedPrice: number;
  modelVersion: string;
  modelAlias: string;
  timestamp: string;
  predictionId: string;
  address: string;
  propertyKey: string;
  warnings: string[];
}

export function PredictionResultCard({
  predictedPrice,
  modelVersion,
  modelAlias,
  timestamp,
  predictionId,
  address,
  propertyKey,
  warnings,
}: PredictionResultCardProps) {
  return (
    <div className="card">
      <h2>Prediction Result</h2>
      <div className="result-price">{formatCurrency(predictedPrice)}</div>
      <p>
        Model: <span className="badge">{modelVersion}</span> ({modelAlias})
      </p>
      <p>Timestamp: {formatDate(timestamp)}</p>
      <p>Address: {address}</p>
      <p>Property key: <code>{propertyKey}</code></p>
      <p>Prediction ID: <code>{predictionId}</code></p>
      {warnings.map((w) => (
        <div key={w} className="warning">{w}</div>
      ))}
    </div>
  );
}
