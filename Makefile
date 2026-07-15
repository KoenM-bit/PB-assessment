.PHONY: install install-ml install-web dev test test-ml test-web lint lint-ml lint-web seed train promote-challenger clean
.PHONY: verify-databricks databricks-init-catalog databricks-init-prod verify-inference verify-inference-prod
.PHONY: deploy-serving deploy-serving-staging deploy-serving-prod promote-champion promote-to-production bootstrap-production
.PHONY: deploy-netlify deploy-netlify-prod netlify-build
.PHONY: databricks-bundle-deploy databricks-bundle-deploy-prod upload-ml-wheel setup-databricks-repo
.PHONY: remote-databricks

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
	bash scripts/run-e2e-smoke.sh

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
	@echo "Experiment + UC version (no @challenger). Go live: make promote-challenger RUN_ID=<id>"
	cd ml && python -m house_price_ml.models.train --data ../data/sample/listings.csv

promote-challenger:
	@test -n "$(RUN_ID)" || (echo "Usage: make promote-challenger RUN_ID=<mlflow-run-id>"; exit 1)
	chmod +x scripts/promote-challenger.py
	cd ml && python ../scripts/promote-challenger.py --run-id "$(RUN_ID)"

verify-databricks:
	chmod +x scripts/verify-databricks.sh
	./scripts/verify-databricks.sh

verify-inference:
	chmod +x scripts/verify-inference.sh
	./scripts/verify-inference.sh staging

verify-inference-prod:
	chmod +x scripts/verify-inference.sh
	./scripts/verify-inference.sh production --skip-e2e

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

deploy-serving-from-registry:
	chmod +x scripts/deploy-serving.sh
	FROM_REGISTRY=true ./scripts/deploy-serving.sh staging

deploy-serving-prod-from-registry:
	chmod +x scripts/deploy-serving.sh
	FROM_REGISTRY=true ./scripts/deploy-serving.sh production

promote-champion:
	chmod +x scripts/promote-champion.py
	cd ml && python ../scripts/promote-champion.py

promote-to-production:
	chmod +x scripts/promote-to-production.py
	@if [ "$$CONFIRM_PROMOTE" != "yes" ]; then \
		echo "Set CONFIRM_PROMOTE=yes to promote staging @challenger to production @champion."; \
		echo "  CONFIRM_PROMOTE=yes make promote-to-production"; \
		echo "Dry run: cd ml && python ../scripts/promote-to-production.py --dry-run"; \
		exit 1; \
	fi
	cd ml && python ../scripts/promote-to-production.py

bootstrap-production:
	chmod +x scripts/bootstrap-production.sh scripts/deploy-serving.sh
	./scripts/bootstrap-production.sh
fetch-serving-logs:
	chmod +x scripts/fetch-serving-logs.py
	cd ml && python ../scripts/fetch-serving-logs.py

databricks-bundle-deploy:
	chmod +x scripts/databricks-bundle-deploy.sh
	./scripts/databricks-bundle-deploy.sh staging

databricks-bundle-deploy-prod:
	chmod +x scripts/databricks-bundle-deploy.sh
	./scripts/databricks-bundle-deploy.sh prod

upload-ml-wheel:
	chmod +x scripts/upload-ml-wheel.sh
	./scripts/upload-ml-wheel.sh

databricks-staging-pipeline:
	chmod +x scripts/databricks-ci.sh
	./scripts/databricks-ci.sh staging-pipeline staging

databricks-staging-pipeline-deploy:
	chmod +x scripts/databricks-ci.sh
	./scripts/databricks-ci.sh staging-pipeline-deploy staging

databricks-production-pipeline:
	chmod +x scripts/databricks-ci.sh
	CONFIRM_PROMOTE=yes ./scripts/databricks-ci.sh production-pipeline prod

setup-databricks-repo:
	chmod +x scripts/setup-databricks-repo.sh
	./scripts/setup-databricks-repo.sh

# Trigger GitHub Actions Databricks workflow (requires: gh auth login)
TARGET ?= staging
remote-databricks:
	@test -n "$(CMD)" || (echo "Usage: make remote-databricks CMD=staging-pipeline [TARGET=staging]"; exit 1)
	gh workflow run databricks.yml -f command=$(CMD) -f target=$(TARGET) \
		$(if $(filter promote-to-production production-pipeline,$(CMD)),-f confirm_promote=yes,)

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

setup-netlify-production:
	chmod +x scripts/setup-netlify-production.sh
	./scripts/setup-netlify-production.sh

clean:
	rm -rf apps/web/dist ml/.pytest_cache ml/.mypy_cache ml/.ruff_cache ml/artifacts
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
