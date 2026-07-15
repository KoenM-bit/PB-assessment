# Deployment

See [configuration.md](configuration.md) for the full environment matrix and precedence rules.

## Environments

| Variable | Local | Staging | Production |
|----------|-------|---------|------------|
| `APP_ENV` | local | staging | production |
| `DATABRICKS_CATALOG` | house_price_staging | house_price_staging | house_price_prod |
| `MODEL_ALIAS` | challenger | challenger | champion |
| `USE_MOCK_DATABRICKS` | true | false | false |

**Source of truth:** deployed Netlify apps use hardcoded catalog/endpoint/alias in `config.ts` (see [configuration.md](configuration.md)).

## Netlify Deployment

### Staging
- Triggered on merge to `staging` branch
- Deploys frontend + functions to staging context
- Points to challenger model

### Production
- Triggered on merge to `master` (manual approval gate optional)
- Points to champion model

### Configuration

Set **once** in Netlify dashboard (shared across staging + production on free tier):
- `DATABRICKS_HOST`
- `DATABRICKS_TOKEN`
- `DATABRICKS_SQL_WAREHOUSE_ID`
- `DEMO_WRITE_TOKEN`
- `USE_MOCK_DATABRICKS=false`

Staging vs production **endpoint and catalog** are chosen automatically:
- `netlify.toml` sets `APP_ENV` per branch/context
- `netlify/functions/_shared/config.ts` maps `APP_ENV` → serving endpoint + catalog

Details: [configuration.md](configuration.md).

## Databricks Deployment

```bash
cd databricks
databricks bundle deploy -t staging
databricks bundle deploy -t prod  # requires approval
```

Deploys:
- SQL schemas and tables
- Workflow jobs (ETL, training, evaluation, monitoring)
- Serving endpoint configuration

## CI/CD Workflows

| Workflow | Trigger | Actions |
|----------|---------|---------|
| `ci.yml` | Pull request | Lint, typecheck, test |
| `databricks.yml` | Push to staging (ml/databricks) + manual | Bundle deploy, ML pipeline, serving |
| `deploy-staging.yml` | Push to staging | E2E smoke test after Netlify deploy |
| `deploy-production.yml` | Manual dispatch | Production verification gate |

See [enterprise-workflow.md](enterprise-workflow.md) for the full online development model.

## Production Promotion Checklist

- [ ] All CI tests pass
- [ ] Candidate model meets quality gates
- [ ] Staging smoke test passes
- [ ] Monitoring queries return data
- [ ] Rollback target (`previous_champion`) identified
- [ ] Manual approval granted

## Rollback Procedure

**Automatic:** `make deploy-serving-from-registry` rolls back the serving endpoint and
registry alias (`@challenger` / `@champion`) if post-deploy inference verification fails.

While the primary endpoint is deploying or unavailable, the Netlify API automatically tries
the **peer environment** serving endpoint (staging → production, production → staging)
before falling back to the business baseline.

**Manual:**

1. In MLflow registry, set `champion` alias to `previous_champion` version
2. Verify serving endpoint serves rolled-back model
3. Monitor `gold.serving_metrics` for error rate
4. No frontend or API code changes needed

## Demo vs Production

| Aspect | Demo | Production |
|--------|------|------------|
| Data processing | Pandas | PySpark at scale |
| Auth | Shared bearer token | OAuth / service principals |
| Drift detection | Simple stats | PSI, KS, Evidently |
| Business KPIs | Placeholder interfaces | Full analytics |
| Auto-retrain | Disabled | Scheduled with gates |
| Infrastructure | Netlify Functions | API gateway + K8s optional |

## Security in Production

- Replace personal access tokens with Databricks service principals
- Enable Netlify access control for admin pages
- Use Unity Catalog row/column masking for PII
- Audit logging for write operations
- Rate limiting on predict endpoint

## Local Development

```bash
cp .env.example .env
make install
make seed
make gold-export
make train
make dev-full   # Netlify dev with functions
```

Mock mode (`USE_MOCK_DATABRICKS=true`) runs without Databricks.
