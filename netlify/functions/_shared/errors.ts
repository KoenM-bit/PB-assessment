import type { ApiEnvelope } from "./schemas.js";
import type { HandlerResponse } from "@netlify/functions";

export class ApiError extends Error {
  constructor(
    public statusCode: number,
    public code: string,
    message: string,
  ) {
    super(message);
  }
}

export function errorResponse(
  statusCode: number,
  code: string,
  message: string,
  requestId: string,
): HandlerResponse {
  const body: ApiEnvelope<null> = {
    data: null,
    error: { code, message },
    meta: { request_id: requestId, timestamp: new Date().toISOString() },
  };
  return {
    statusCode,
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  };
}

export function successResponse<T>(
  data: T,
  requestId: string,
  status = 200,
): HandlerResponse {
  const body: ApiEnvelope<T> = {
    data,
    error: null,
    meta: { request_id: requestId, timestamp: new Date().toISOString() },
  };
  return {
    statusCode: status,
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  };
}

export function handleError(err: unknown, requestId: string): HandlerResponse {
  if (err instanceof ApiError) {
    return errorResponse(err.statusCode, err.code, err.message, requestId);
  }
  console.error(err);
  return errorResponse(500, "INTERNAL_ERROR", "An unexpected error occurred", requestId);
}
