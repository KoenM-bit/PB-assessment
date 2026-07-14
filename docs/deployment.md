# Deployment

## Environments

| Variable | Local | Staging | Production |
|----------|-------|---------|------------|
| `APP_ENV` | local | staging | production |
| `DATABRICKS_CATALOG` | house_price_staging | house_price_staging | house_price_prod |
| `MODEL_ALIAS` | challenger | challenger | champion |
| `USE_MOCK_DATABRICKS` | true | false | false |

## Netlify Deployment

### Staging
- Triggered on merge to `staging` branch
- Deploys frontend + functions to staging context
- Points to challenger model

### Production
- Triggered on merge to `master` (manual approval gate optional)
- Points to champion model

### Configuration

Set in Netlify dashboard (never in client code):
- `DATABRICKS_HOST`
- `DATABRICKS_TOKEN`
- `DATABRICKS_SERVING_ENDPOINT`
- `DATABRICKS_SQL_WAREHOUSE_ID`
- `DEMO_WRITE_TOKEN`

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
| `pr.yml` | Pull request | Lint, typecheck, test |
| `deploy-staging.yml` | Push to staging | Build, deploy Netlify + Databricks staging, smoke test |
| `deploy-production.yml` | Manual dispatch | Deploy production with approval |

## Production Promotion Checklist

- [ ] All CI tests pass
- [ ] Candidate model meets quality gates
- [ ] Staging smoke test passes
- [ ] Monitoring queries return data
- [ ] Rollback target (`previous_champion`) identified
- [ ] Manual approval granted

## Rollback Procedure

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
make train
make dev-full   # Netlify dev with functions
```

Mock mode (`USE_MOCK_DATABRICKS=true`) runs without Databricks.
