# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- Repository skeleton: uv workspace with the `modelport` SDK and `modelport-platform` packages,
  pinned `versions.env`, `Makefile`, license and dataset credits.
- Development toolbox: `docker compose run --rm toolbox make <target>` runs every target with the
  pinned tools and only Docker on the host (ADR-0019).
- Static CI (lint, types, tests, Helm lint, version consistency) and `scripts/check_versions.py`.
- kind cluster `modelport` with namespaces `platform`, `ray`, `monitoring`.
- Ray image (`deploy/images/ray`, hash-locked), KubeRay operator 1.7.3 and a smoke RayJob.
- `scripts/secrets.sh` (all Secrets, generated once) and PostgreSQL 18 with the `airflow` and `mlflow` databases.
- Airflow image (hash-locked from the 3.4.0 constraints) and chart values: Airflow 3.4.0 on chart 1.22.0 with
  LocalExecutor and SimpleAuthManager.
- MinIO built from source, and a Job that creates the eight S3 buckets idempotently.
- MLflow 3.18 tracking and registry server (PostgreSQL backend, artifacts in MinIO).
- `make up` / `make down` (full and smoke profiles) with a readiness wait, and a weekly CI check that
  `make up` finishes within 15 minutes when the images are present.
- SDK foundations: the error hierarchy with CLI exit codes, every name and path in `modelport.naming`,
  client settings and structured logging.
- `modelport validate DIR`: twelve ordered checks with a fix for each failure, no cluster needed; global
  `--json`, `--debug` and `--version`.
- MLflow pyfunc adapter: a trained model is logged once (code, weights, spec, signature) and loaded the
  same way by serving, batch and evaluation.
- MLflow registry adapter: versions, the three aliases, tags, submission runs and artifacts, with
  retries on network failures.
- Model version lifecycle: states, legal transitions and the ordered alias changes for gate, canary,
  promotion and rollback.
- Deterministic code packaging (`code.zip` with a content hash that is the same on every OS).
- `BaseModel` and `Predictions` (the contract a model implements), loading a model class from its
  folder, and input/output checks against the signature; the MLflow signature built from `model.yaml`.
- `model.yaml` spec (`ModelSpec`) with field-path error messages and its exported JSON Schema.
- Local Ray Train integration test: TorchTrainer with DDP (gloo), a checkpoint per epoch and resume; XGBoostTrainer
  with a checkpoint per boosting round and resume.
