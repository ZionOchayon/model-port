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
- Kubernetes adapter for KubeRay objects (idempotent create, KubeRay status mapping), the training
  RayJob builder, and the RBAC that lets Airflow manage Ray resources.
- Airflow DAGs `ingest_dataset` (deferrable S3 sensor, version or quarantine) and `retrain_on_dataset`
  (retrain subscribed models on a new dataset version); `make ui`; CI DAG tests.
- Ingest: quality checks against a dataset contract, immutable dataset versions (manifest written last)
  and quarantine with a report.
- Airflow DAG `evaluate_and_promote`: the new version and the champion are evaluated on the same
  validation set, the gate decides, and the version becomes `candidate` or `rejected` with reasons;
  `train_model` starts it after registration. Lifecycle and job-timing pushes to a Pushgateway.
- Shared RayCluster `modelport-shared` (workers scale from zero) for short jobs, idempotent job
  submission to it, and the evaluation job that scores a version and times its inference.
- `modelport submit DIR [--spec FILE] [--dataset-version N] [--wait]`: validate, package and start training
  through the Airflow REST API; resubmitting the same code reuses the existing run.
- Evaluation gate rules: an absolute bar for a model's first version; otherwise the primary metric,
  every class's score and latency are checked against the champion, and every failed reason is reported.
- Airflow DAG `train_model`: one RayJob per submission on its own Ray cluster, waited on with a
  rescheduling sensor, then a check that the version was registered.
- Model registration after training: an MLflow version with signature, lineage tags and a drift
  baseline; Population Stability Index for continuous and categorical features.
- Training on Ray Train: the RayJob entrypoint, Torch (DDP over gloo) and XGBoost strategies,
  dataset splits read with Ray Data, and resume from the last checkpoint when a RayJob is rerun.
- Platform settings from the environment, the platform ports, and the S3 (MinIO) adapter; a test keeps the
  pure core free of I/O libraries.
- `examples/datasets/prepare.py`: downloads Fashion-MNIST, EuroSAT or Covertype, converts them to the
  ingest layout and uploads training files, corrupt files or batch input.
- Example model 2: a Covertype tree-cover classifier (XGBoost on Ray Train), with its own dependencies.
- Example model 1: an image classifier (ResNet-18, PyTorch DDP) with EuroSAT and Fashion-MNIST spec files.
- Training helpers for `train_loop`: `modelport.report()` (per-epoch metrics and checkpoint, rank 0 only)
  and `modelport.latest_artifacts()` (resume).
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
