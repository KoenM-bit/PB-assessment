# House Price Prediction — Technical Assessment

A production-inspired, demo-sized house price prediction application demonstrating the complete ML system lifecycle.

## Architecture

- **Frontend:** React + TypeScript + Vite, hosted on Netlify
- **API:** Netlify Functions (server-side Databricks credentials)
- **ML Platform:** Databricks (Delta medallion, Workflows, MLflow, Model Serving)

## Quick Start

```bash
# Install dependencies
make install

# Generate sample data
make seed

# Train model locally
make train

# Run all tests
make test

# Start frontend dev server
make dev

# Start full stack with Netlify dev (functions + frontend)
make dev-full
```

## Environment

Copy `.env.example` to `.env` and configure Databricks credentials for non-mock mode:

```bash
cp .env.example .env
```

Set `USE_MOCK_DATABRICKS=true` for local development without a Databricks workspace.

## Project Structure

| Path | Purpose |
|------|---------|
| `apps/web/` | React dashboard |
| `netlify/functions/` | Serverless API layer |
| `ml/` | Python ML package (features, training, serving) |
| `databricks/` | SQL DDL, notebooks, workflows, asset bundle |
| `docs/` | Architecture and operational documentation |
| `tests/` | Integration and E2E tests |

## Documentation

- [Architecture](docs/architecture.md)
- [Data Model](docs/data_model.md)
- [Model Lifecycle](docs/model_lifecycle.md)
- [Monitoring](docs/monitoring.md)
- [Testing](docs/testing.md)
- [Deployment](docs/deployment.md)
- [Databricks setup](docs/databricks-setup.md)
- [Security & Rollback](docs/security.md)

## Environments

| Environment | Branch | Model Alias | Catalog |
|-------------|--------|-------------|---------|
| Local | — | challenger | house_price_staging |
| Staging | `staging` | challenger | house_price_staging |
| Production | `main` | champion | house_price_prod |

## Demo vs Production

This repository prioritises clarity over enterprise complexity. See [deployment docs](docs/deployment.md) for deliberate simplifications and scaling paths.
