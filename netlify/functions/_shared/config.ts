export interface ServingTarget {
  servingEndpoint: string;
  catalog: string;
  modelAlias: string;
}

export interface AppConfig {
  appEnv: string;
  databricksHost: string;
  databricksToken: string;
  servingEndpoint: string;
  sqlWarehouseId: string;
  catalog: string;
  schema: string;
  modelAlias: string;
  peerServingTarget: ServingTarget | null;
  minEvaluationSampleSize: number;
  demoWriteToken: string;
  servingTimeoutMs: number;
  sqlMaxWaitMs: number;
  useMockDatabricks: boolean;
}

/** Deployed targets per APP_ENV — avoids per-context vars in Netlify UI (free tier). */
function deployedDatabricksTarget(appEnv: string): ServingTarget | null {
  if (appEnv === "production") {
    return {
      servingEndpoint: "house-price-serving-prod",
      catalog: "house_price_prod",
      modelAlias: "champion",
    };
  }
  if (appEnv === "staging") {
    return {
      servingEndpoint: "house-price-serving",
      catalog: "house_price_staging",
      modelAlias: "challenger",
    };
  }
  return null;
}

/** Cross-environment serving used when primary endpoint is deploying or unavailable. */
function peerServingTarget(appEnv: string): ServingTarget | null {
  if (appEnv === "staging") {
    return {
      servingEndpoint:
        process.env.PEER_SERVING_ENDPOINT || "house-price-serving-prod",
      catalog: process.env.PEER_DATABRICKS_CATALOG || "house_price_prod",
      modelAlias: process.env.PEER_MODEL_ALIAS || "champion",
    };
  }
  if (appEnv === "production") {
    return {
      servingEndpoint: process.env.PEER_SERVING_ENDPOINT || "house-price-serving",
      catalog: process.env.PEER_DATABRICKS_CATALOG || "house_price_staging",
      modelAlias: process.env.PEER_MODEL_ALIAS || "challenger",
    };
  }
  return null;
}

function peerFallbackEnabled(appEnv: string): boolean {
  if (appEnv !== "staging" && appEnv !== "production") {
    return false;
  }
  return process.env.ENABLE_PEER_SERVING_FALLBACK !== "false";
}

export function getConfig(): AppConfig {
  const appEnv = process.env.APP_ENV || "local";
  const deployed = deployedDatabricksTarget(appEnv);
  const peer = peerFallbackEnabled(appEnv) ? peerServingTarget(appEnv) : null;

  return {
    appEnv,
    databricksHost: process.env.DATABRICKS_HOST || "",
    databricksToken: process.env.DATABRICKS_TOKEN || "",
    servingEndpoint:
      deployed?.servingEndpoint || process.env.DATABRICKS_SERVING_ENDPOINT || "house-price-serving",
    sqlWarehouseId: process.env.DATABRICKS_SQL_WAREHOUSE_ID || "",
    catalog: deployed?.catalog || process.env.DATABRICKS_CATALOG || "house_price_staging",
    schema: process.env.DATABRICKS_SCHEMA || "gold",
    modelAlias: deployed?.modelAlias || process.env.MODEL_ALIAS || "challenger",
    peerServingTarget: peer,
    minEvaluationSampleSize: parseInt(process.env.MIN_EVALUATION_SAMPLE_SIZE || "30", 10),
    demoWriteToken: process.env.DEMO_WRITE_TOKEN || "",
    servingTimeoutMs: parseInt(process.env.SERVING_TIMEOUT_MS || "30000", 10),
    sqlMaxWaitMs: parseInt(process.env.SQL_MAX_WAIT_MS || "25000", 10),
    useMockDatabricks: process.env.USE_MOCK_DATABRICKS !== "false",
  };
}
