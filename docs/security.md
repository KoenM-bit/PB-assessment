# Security

## Demo Implementation

| Control | Implementation |
|---------|----------------|
| Databricks credentials | Netlify environment variables only |
| Write protection | `DEMO_WRITE_TOKEN` bearer auth on `POST /api/actual-sales` |
| Input validation | Zod (API) + Pydantic/sklearn (ML) |
| Staging/production isolation | Separate Unity Catalog names |
| PII logging | No postcode logged in prediction warnings |

## Production Recommendations

### Authentication

Replace shared bearer token with:
- Netlify Identity or Auth0 for admin pages
- Databricks service principal for API → Databricks calls
- OAuth machine-to-machine for CI/CD deployments

### Secrets Management

- Store `DATABRICKS_TOKEN` in Netlify encrypted env vars
- Rotate tokens quarterly
- Use separate tokens per environment
- Never commit `.env` files

### Network

- Restrict Databricks serving endpoint to Netlify egress IPs (if supported)
- Enable Databricks IP access lists on SQL warehouse
- Use private link for production workspaces

### Data

- Unity Catalog row filters for sensitive fields
- Mask postcodes in monitoring dashboards
- Audit trail on `gold.actual_sales` writes

## Rollback Runbook

### Model Rollback (< 5 minutes)

1. Identify `previous_champion` version in MLflow registry
2. Set `champion` alias to that version:
   ```bash
   mlflow models set-alias --name house_price_model --alias champion --version <N>
   ```
3. Verify serving endpoint health
4. Monitor `gold.serving_metrics` for 15 minutes

### API Rollback

1. In Netlify, redeploy previous production deploy
2. Or revert git commit on `main` and trigger production workflow

### Data Rollback

Delta tables support time travel:
```sql
RESTORE TABLE gold.predictions TO VERSION AS OF <version>
```

Use only for critical data corruption — predictions are append-only by design.

## Incident Response

1. Check monitoring dashboard for error rate spike
2. If serving fails → automatic fallback to business baseline (logged as `is_fallback=true`)
3. If baseline fails → API returns 502 with structured error
4. Escalate to ML team if fallback rate > 5%

## Demo vs Production Gap

| Area | Demo | Production |
|------|------|------------|
| Auth | Shared token | OAuth + RBAC |
| Secrets | `.env` file | Vault / KMS |
| Audit | Console logs | SIEM integration |
| Rate limiting | None | API gateway throttling |
| Encryption | HTTPS only | TLS + at-rest encryption |
