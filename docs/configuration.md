# Configuration

Single reference for **where each setting is defined** across local, staging, and production. See also [deployment.md](deployment.md) for deploy procedures.

## Environment identity

| `APP_ENV` | Trigger | Databricks catalog | Serving endpoint | Model alias |
|-----------|---------|-------------------|------------------|-------------|
| `local` | `.env` | `DATABRICKS_CATALOG` (default `house_price_staging`) | `DATABRICKS_SERVING_ENDPOINT` env | `MODEL_ALIAS` env |
| `staging` | Netlify branch `staging` | `house_price_staging` | `house-price-serving` | `challenger` |
| `production` | Netlify branch `master` | `house_price_prod` | `house-price-serving-prod` | `champion` |

Staging and production values for catalog, endpoint, and alias are **hardcoded** in [`netlify/functions/_shared/config.ts`](../netlify/functions/_shared/config.ts) (`deployedDatabricksTarget()`). [`netlify.toml`](../netlify.toml) context blocks document the same values as a backstop.

---

## Source of truth by layer

| Concern | Local | Staging | Production | Defined in |
|---------|-------|---------|------------|------------|
| App environment | `.env` `APP_ENV=local` | `APP_ENV=staging` | `APP_ENV=production` | `.env` / `netlify.toml` |
| UC catalog (API) | `.env` or mock | `house_price_staging` | `house_price_prod` | `config.ts` when deployed |
| UC catalog (Databricks jobs) | N/A | `house_price_staging` | `house_price_prod` | [`databricks/databricks.yml`](../databricks/databricks.yml) bundle targets |
| UC catalog (Python scripts) | `.env` → `Settings` | CI secrets + `.env` | promote/deploy scripts | [`ml/src/house_price_ml/config/settings.py`](../ml/src/house_price_ml/config/settings.py) |
| Serving endpoint | `.env` | `house-price-serving` | `house-price-serving-prod` | `config.ts` / `netlify.toml` |
| Model alias (live) | `.env` | `@challenger` | `@champion` | MLflow registry + `deploy-serving` |
| Databricks credentials | `.env` | Netlify dashboard (shared) | same | never in frontend |
| SQL warehouse ID | `.env` | Netlify dashboard | same | Netlify env |
| MLflow experiment | `/Shared/house_price_prediction` | same | same | `Settings` / scripts |
| MLflow tracking URI | `databricks` or sqlite (tests) | `databricks` | `databricks` | scripts set at runtime |
| MLflow registry URI | `databricks-uc` | same | same | promotion scripts |
| Mock Databricks | `USE_MOCK_DATABRICKS=true` (default) | `false` | `false` | `.env` / Netlify |
| Peer serving fallback | off | on (default) | on (default) | `config.ts` + `ENABLE_PEER_SERVING_FALLBACK` |
| Min evaluation sample | `30` | `30` | `30` | `MIN_EVALUATION_SAMPLE_SIZE` |
| Write protection token | `.env` | Netlify dashboard | same | `DEMO_WRITE_TOKEN` |

---

## Precedence rules

### Netlify API (deployed staging / production)

When `APP_ENV` is `staging` or `production`, [`deployedDatabricksTarget()`](../netlify/functions/_shared/config.ts) **wins** for:

- `catalog`
- `servingEndpoint`
- `modelAlias`

Do not rely on per-context Netlify UI variables for catalog/endpoint on the free tier — code mapping is the source of truth.

`DATABRICKS_CATALOG` / `DATABRICKS_SERVING_ENDPOINT` in `netlify.toml` are documentation only when `APP_ENV` is set.

### Local development

1. Repo-root `.env` (see [`.env.example`](../.env.example))
2. `Settings` defaults in Python
3. `getConfig()` fallbacks in TypeScript (`USE_MOCK_DATABRICKS` defaults to mock unless explicitly `false`)

### Databricks jobs

- Bundle target (`staging` / `prod`) sets `var.catalog` and `var.model_alias`
- Passed to notebooks as widgets (`catalog`, `git_commit`)
- Jobs do **not** read Netlify configuration

### Promotion and deploy scripts

- `MLFLOW_TRACKING_URI=databricks` and `MLFLOW_REGISTRY_URI=databricks-uc` set in script
- Catalog from `DATABRICKS_CATALOG` environment variable
- Endpoint from script argument or `DATABRICKS_SERVING_ENDPOINT`

---

## What to set where

### Netlify dashboard (shared across contexts)

Set once:

| Variable | Purpose |
|----------|---------|
| `DATABRICKS_HOST` | Workspace URL |
| `DATABRICKS_TOKEN` | API token (use service principal in production) |
| `DATABRICKS_SQL_WAREHOUSE_ID` | SQL warehouse for monitoring queries |
| `DEMO_WRITE_TOKEN` | Protects `POST /api/actual-sales` |
| `USE_MOCK_DATABRICKS=false` | Enable real Databricks calls |

Do **not** need separate staging/production values for catalog or endpoint — `APP_ENV` routing handles that.

### GitHub Actions secrets

Same Databricks credentials as local `.env`:

- `DATABRICKS_HOST`
- `DATABRICKS_TOKEN`
- `DATABRICKS_SQL_WAREHOUSE_ID`

Workflows pass `git_commit` and bundle target to Databricks jobs.

### Local `.env`

For `make train`, `make deploy-serving`, and other scripts:

```bash
cp .env.example .env
```

Key locals:

| Variable | Typical local value |
|----------|---------------------|
| `APP_ENV` | `local` |
| `USE_MOCK_DATABRICKS` | `true` (until Databricks configured) |
| `DATABRICKS_CATALOG` | `house_price_staging` |
| `MODEL_ALIAS` | `challenger` |

---

## Known default differences

| Setting | Python `Settings` default | Netlify / `.env.example` | Applies when |
|---------|---------------------------|--------------------------|--------------|
| `serving_timeout_ms` | `10000` | `30000` | Netlify predict function uses `config.ts` (30s); local Python scripts use `Settings` unless env overrides |
| `use_mock_databricks` | `true` | `false` in deployed contexts | Local mock vs live API |

These are intentional: local scripts fail fast; deployed API allows more headroom for cold-start serving.

---

## Optional overrides

| Variable | Default | Purpose |
|----------|---------|---------|
| `ENABLE_PEER_SERVING_FALLBACK` | `true` (staging/prod) | Cross-env serving when primary endpoint unavailable |
| `PEER_SERVING_ENDPOINT` | peer env default | Override peer endpoint name |
| `PEER_DATABRICKS_CATALOG` | peer env default | Override peer catalog |
| `PEER_MODEL_ALIAS` | peer env default | Override peer alias |
| `REGISTER_UC_MODEL` | off | Opt-in UC registration from notebooks (`1`/`true`/`yes`) |

---

## Related docs

- [deployment.md](deployment.md) — deploy and rollback procedures
- [enterprise-workflow.md](enterprise-workflow.md) — daily development and promotion flow
- [data_model.md](data_model.md) — training data path (`gold.training_frame`)
