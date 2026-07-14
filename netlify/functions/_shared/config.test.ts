import { afterEach, describe, expect, it } from "vitest";
import { getConfig } from "./config.js";

const ENV_KEYS = [
  "APP_ENV",
  "DATABRICKS_SERVING_ENDPOINT",
  "DATABRICKS_CATALOG",
  "MODEL_ALIAS",
] as const;

function snapshotEnv(): Record<string, string | undefined> {
  return Object.fromEntries(ENV_KEYS.map((key) => [key, process.env[key]]));
}

function restoreEnv(snapshot: Record<string, string | undefined>): void {
  for (const key of ENV_KEYS) {
    const value = snapshot[key];
    if (value === undefined) delete process.env[key];
    else process.env[key] = value;
  }
}

describe("getConfig", () => {
  const saved = snapshotEnv();

  afterEach(() => restoreEnv(saved));

  it("routes production deploys to prod catalog and endpoint", () => {
    process.env.APP_ENV = "production";
    process.env.DATABRICKS_SERVING_ENDPOINT = "house-price-serving";
    process.env.DATABRICKS_CATALOG = "house_price_staging";
    process.env.MODEL_ALIAS = "champion";

    const config = getConfig();
    expect(config.servingEndpoint).toBe("house-price-serving-prod");
    expect(config.catalog).toBe("house_price_prod");
    expect(config.modelAlias).toBe("champion");
  });

  it("routes staging deploys to staging catalog and endpoint", () => {
    process.env.APP_ENV = "staging";
    process.env.DATABRICKS_SERVING_ENDPOINT = "house-price-serving-prod";
    process.env.DATABRICKS_CATALOG = "house_price_prod";

    const config = getConfig();
    expect(config.servingEndpoint).toBe("house-price-serving");
    expect(config.catalog).toBe("house_price_staging");
    expect(config.modelAlias).toBe("challenger");
  });

  it("respects .env overrides for local development", () => {
    process.env.APP_ENV = "local";
    process.env.DATABRICKS_SERVING_ENDPOINT = "custom-endpoint";
    process.env.DATABRICKS_CATALOG = "custom_catalog";
    process.env.MODEL_ALIAS = "challenger";

    const config = getConfig();
    expect(config.servingEndpoint).toBe("custom-endpoint");
    expect(config.catalog).toBe("custom_catalog");
  });
});
