export interface AppConfig {
  appEnv: string;
  databricksHost: string;
  databricksToken: string;
  servingEndpoint: string;
  sqlWarehouseId: string;
  catalog: string;
  schema: string;
  modelAlias: string;
  minEvaluationSampleSize: number;
  demoWriteToken: string;
  servingTimeoutMs: number;
  sqlMaxWaitMs: number;
  useMockDatabricks: boolean;
}

/** Deployed targets per APP_ENV — avoids per-context vars in Netlify UI (free tier). */
function deployedDatabricksTarget(appEnv: string): Pick<AppConfig, "servingEndpoint" | "catalog" | "modelAlias"> | null {
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

export function getConfig(): AppConfig {
  const appEnv = process.env.APP_ENV || "local";
  const deployed = deployedDatabricksTarget(appEnv);

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
    minEvaluationSampleSize: parseInt(process.env.MIN_EVALUATION_SAMPLE_SIZE || "30", 10),
    demoWriteToken: process.env.DEMO_WRITE_TOKEN || "",
    servingTimeoutMs: parseInt(process.env.SERVING_TIMEOUT_MS || "30000", 10),
    sqlMaxWaitMs: parseInt(process.env.SQL_MAX_WAIT_MS || "25000", 10),
    useMockDatabricks: process.env.USE_MOCK_DATABRICKS !== "false",
  };
}
