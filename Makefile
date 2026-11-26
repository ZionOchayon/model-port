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
	@echo "make up: not implemented yet"; exit 1

down: ## Delete the cluster and local state
	@echo "make down: not implemented yet"; exit 1

images: ## Build and load the three modelport images
	@echo "make images: not implemented yet"; exit 1

ui: ## Port-forward the UIs and print their URLs
	@echo "make ui: not implemented yet"; exit 1

demo: ## Onboard both example models end to end, then inject drift
	@echo "make demo: not implemented yet"; exit 1

verify: ## Check runs from their artifacts (MLflow, RayJob state, S3, Prometheus)
	@echo "make verify: not implemented yet"; exit 1

bench: ## FR-13 measurements and failure drills; updates docs/results and README
	@echo "make bench: not implemented yet"; exit 1

metrics-check: ## Every Prometheus target is up and every catalogue metric has data
	@echo "make metrics-check: not implemented yet"; exit 1

test: ## Unit, contract and local integration tests (no cluster)
	@uv run pytest

lint: ## ruff check + format check
	@uv run ruff check . && uv run ruff format --check .

typecheck: ## mypy --strict
	@uv run mypy

rules-test: ## promtool unit tests for the PrometheusRule
	@echo "make rules-test: not implemented yet"; exit 1
