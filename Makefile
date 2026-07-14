.PHONY: install install-ml install-web dev test test-ml test-web lint lint-ml lint-web seed train clean
.PHONY: verify-databricks databricks-init-catalog databricks-init-prod
.PHONY: deploy-serving deploy-serving-staging deploy-serving-prod promote-champion bootstrap-production
.PHONY: deploy-netlify deploy-netlify-prod netlify-build

install: install-ml install-web

install-ml:
	cd ml && pip install -e ".[dev]"

install-web:
	cd apps/web && npm install
	cd netlify/functions && npm install

dev:
	@echo "Tip: use 'make dev-full' to start frontend + API together."
	cd apps/web && VITE_API_PROXY_TARGET=http://localhost:9999 npm run dev

dev-full:
	chmod +x scripts/dev-full.sh scripts/dev-web.sh
	./scripts/dev-full.sh

dev-netlify:
	cd "$(CURDIR)" && netlify dev --functions netlify/functions

test: test-ml test-web test-integration

test-ml:
	cd ml && pytest -v

test-web:
	cd apps/web && npm test -- --run
	cd netlify/functions && npm test

test-integration:
	pytest tests/integration -v

test-e2e:
	pytest tests/e2e -v

lint: lint-ml lint-web

lint-ml:
	cd ml && ruff check src tests
	cd ml && mypy src

lint-web:
	cd apps/web && npm run lint
	cd apps/web && npm run typecheck

seed:
	cd ml && python -m house_price_ml.data.synthetic --output ../data/sample/listings.csv --rows 500

train:
	cd ml && python -m house_price_ml.models.train --data ../data/sample/listings.csv

verify-databricks:
	chmod +x scripts/verify-databricks.sh
	./scripts/verify-databricks.sh

databricks-init-catalog:
	chmod +x scripts/databricks-init-catalog.sh
	./scripts/databricks-init-catalog.sh $(CATALOG)

databricks-init-prod:
	$(MAKE) databricks-init-catalog CATALOG=house_price_prod

deploy-serving:
	chmod +x scripts/deploy-serving.sh
	./scripts/deploy-serving.sh staging

deploy-serving-staging: deploy-serving

deploy-serving-prod:
	chmod +x scripts/deploy-serving.sh
	./scripts/deploy-serving.sh production

promote-champion:
	chmod +x scripts/promote-champion.py
	cd ml && python ../scripts/promote-champion.py

bootstrap-production:
	chmod +x scripts/bootstrap-production.sh scripts/deploy-serving.sh
	./scripts/bootstrap-production.sh
fetch-serving-logs:
	chmod +x scripts/fetch-serving-logs.py
	cd ml && python ../scripts/fetch-serving-logs.py

netlify-build:
	chmod +x scripts/netlify-build.sh
	./scripts/netlify-build.sh

deploy-netlify:
	chmod +x scripts/netlify-deploy.sh
	./scripts/netlify-deploy.sh

deploy-netlify-prod:
	chmod +x scripts/netlify-deploy.sh
	./scripts/netlify-deploy.sh --prod --context staging

setup-github-protection:
	chmod +x scripts/setup-github-branch-protection.sh
	./scripts/setup-github-branch-protection.sh

setup-netlify-previews:
	chmod +x scripts/setup-netlify-previews.sh
	./scripts/setup-netlify-previews.sh

clean:
	rm -rf apps/web/dist ml/.pytest_cache ml/.mypy_cache ml/.ruff_cache ml/artifacts
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
