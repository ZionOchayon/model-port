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
