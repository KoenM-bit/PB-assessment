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
  useMockDatabricks: boolean;
}

export function getConfig(): AppConfig {
  return {
    appEnv: process.env.APP_ENV || "local",
    databricksHost: process.env.DATABRICKS_HOST || "",
    databricksToken: process.env.DATABRICKS_TOKEN || "",
    servingEndpoint: process.env.DATABRICKS_SERVING_ENDPOINT || "house-price-serving",
    sqlWarehouseId: process.env.DATABRICKS_SQL_WAREHOUSE_ID || "",
    catalog: process.env.DATABRICKS_CATALOG || "house_price_staging",
    schema: process.env.DATABRICKS_SCHEMA || "gold",
    modelAlias: process.env.MODEL_ALIAS || "challenger",
    minEvaluationSampleSize: parseInt(process.env.MIN_EVALUATION_SAMPLE_SIZE || "30", 10),
    demoWriteToken: process.env.DEMO_WRITE_TOKEN || "",
    servingTimeoutMs: parseInt(process.env.SERVING_TIMEOUT_MS || "10000", 10),
    useMockDatabricks: process.env.USE_MOCK_DATABRICKS !== "false",
  };
}
