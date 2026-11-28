# modelport: the single interface. Targets are documented in README "Commands".
# Inside the dev toolbox: docker compose run --rm toolbox make <target> (ADR-0019).
SHELL := /bin/bash
include versions.env
export

PROFILE ?= full
MODEL ?= fashion-classifier
UP_TIMEOUT ?= 1500
REPS ?= 5

.PHONY: help up down ui demo verify bench metrics-check test lint typecheck rules-test images

help: ## List targets
	@grep -E '^[a-z-]+:.*## ' Makefile | sed 's/:.*## /\t/'

up: ## Create the kind cluster and install everything (PROFILE=full|smoke)
	@PROFILE=$(PROFILE) UP_TIMEOUT=$(UP_TIMEOUT) bash scripts/up.sh

down: ## Delete the cluster and local state
	@bash scripts/down.sh

images: ## Build and load the three modelport images
	@bash scripts/images.sh

ui: ## Port-forward the UIs and print their URLs
	@bash scripts/ui.sh

demo: ## Onboard both example models end to end, then inject drift
	@echo "make demo: not implemented yet"; exit 1

verify: ## Check runs from their artifacts (MLflow, RayJob state, S3, Prometheus)
	@echo "make verify: not implemented yet"; exit 1

bench: ## FR-13 measurements and failure drills; updates docs/results and README (REPS=5)
	@bash scripts/ui.sh && set -a && . .modelport/credentials.env && set +a && uv run python bench/bench.py --reps $(REPS)

metrics-check: ## Every Prometheus target is up and every catalogue metric has data
	@uv run python scripts/metrics_check.py --profile $(PROFILE)

test: ## Unit, contract and local integration tests (no cluster)
	@uv run pytest

lint: ## ruff check + format check
	@uv run ruff check . && uv run ruff format --check .

typecheck: ## mypy --strict
	@uv run mypy

# promtool runs natively when on PATH (always inside the toolbox, ADR-0019), else from the pinned image.
PROMTOOL := $(shell command -v promtool >/dev/null 2>&1 && echo promtool || echo 'docker run --rm -v "$(CURDIR)/deploy/monitoring:/w" -w /w --entrypoint promtool $(PROMETHEUS_IMAGE)')

rules-test: ## promtool unit tests for the PrometheusRule
	@uv run python scripts/extract_rules.py deploy/monitoring/prometheus-rules.yaml deploy/monitoring/.rules.generated.yaml
	@cd deploy/monitoring && $(PROMTOOL) test rules rules-tests.yaml
