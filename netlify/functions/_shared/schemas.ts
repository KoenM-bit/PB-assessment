import { z } from "zod";

const dutchPostcode = z
  .string()
  .regex(/^\d{4}\s?[A-Za-z]{2}$/, "Expected Dutch postcode (e.g. 3512 AB)");

export const predictRequestSchema = z.object({
  address: z.string().trim().min(3).max(200),
  postcode: z.preprocess(
    (value) => (typeof value === "string" && value.trim() === "" ? undefined : value),
    dutchPostcode.optional(),
  ),
  listing_id: z.string().uuid().optional(),
  surface_area: z.number().positive(),
  number_of_rooms: z.number().int().positive(),
  number_of_bedrooms: z.number().int().min(0),
  build_year: z.number().int().min(1800).max(new Date().getFullYear()),
  energy_label: z.enum(["A++", "A+", "A", "B", "C", "D", "E", "F", "G"]),
  property_type: z.enum([
    "apartment",
    "terraced_house",
    "semi_detached",
    "detached",
    "bungalow",
  ]),
  garden: z.boolean(),
  region: z.enum([
    "Amsterdam",
    "Rotterdam",
    "Utrecht",
    "The Hague",
    "Eindhoven",
    "Groningen",
    "Maastricht",
    "Nijmegen",
  ]),
  latitude: z.number().min(50.75).max(53.55),
  longitude: z.number().min(3.35).max(7.22),
  prediction_date: z.string().regex(/^\d{4}-\d{2}-\d{2}$/).optional(),
});

export const actualSaleRequestSchema = z.object({
  prediction_id: z.string().uuid(),
  actual_sale_price: z.number().positive(),
  sale_date: z.string().regex(/^\d{4}-\d{2}-\d{2}$/),
});

export type PredictRequest = z.infer<typeof predictRequestSchema>;
export type ModelPredictPayload = Omit<PredictRequest, "address" | "postcode" | "listing_id">;
export type ActualSaleRequest = z.infer<typeof actualSaleRequestSchema>;

/** Strip identity fields before sending to the ML serving endpoint. */
export function toModelPayload(request: PredictRequest): ModelPredictPayload {
  const { address: _address, postcode: _postcode, listing_id: _listingId, ...modelFields } = request;
  return {
    ...modelFields,
    prediction_date: modelFields.prediction_date || new Date().toISOString().slice(0, 10),
  };
}

/** Normalized key for linking predictions to future sales data. */
export function buildPropertyKey(address: string, postcode?: string, region?: string): string {
  return [address, postcode, region]
    .filter((part): part is string => Boolean(part?.trim()))
    .map((part) => part.trim().toLowerCase())
    .join("|")
    .replace(/\s+/g, " ");
}

export interface PredictionResponse {
  prediction_id: string;
  predicted_price: number;
  model_name: string;
  model_version: string;
  model_alias: string;
  prediction_timestamp: string;
  warnings: string[];
}

export interface ApiEnvelope<T> {
  data: T | null;
  error: { code: string; message: string } | null;
  meta: { request_id: string; timestamp: string };
}
