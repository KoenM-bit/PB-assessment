import { formatCurrency } from "./format";

export function formatFeatureValue(feature: string, value: number): string {
  if (feature.includes("price") || feature.includes("surface_x")) {
    return formatCurrency(value);
  }
  if (feature === "dist_to_city_centre_km") {
    return `${value.toFixed(1)} km`;
  }
  if (feature === "surface_area") {
    return `${value.toFixed(0)} m²`;
  }
  return value.toFixed(1);
}

export function featureLabel(feature: string): string {
  return feature.replace(/_/g, " ");
}
