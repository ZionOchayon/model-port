# modelport

[![CI](https://github.com/ZionOchayon/model-port/actions/workflows/ci.yml/badge.svg)](https://github.com/ZionOchayon/model-port/actions/workflows/ci.yml) [![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

A data scientist puts a model into production with one command, on a local Kubernetes cluster.

modelport is an open-source ML platform on kind: distributed training on Ray, dataset versioning with quality checks, and an evaluation gate, all driven by `modelport submit`. Serving with a canary (v0.2) and drift monitoring (v0.3) are next; see [Scope](#scope-of-v01).

## Quickstart

You need Docker with 16 GB of memory for the engine and [uv](https://docs.astral.sh/uv/). kind, kubectl and Helm come with the dev toolbox (or install them with `scripts/install-tools.sh` on Linux).

```bash
# 1. bring the platform up (once; the first run builds the images)
docker network create kind
docker compose run --rm toolbox make up
docker compose run --rm --service-ports toolbox make ui      # keep this running: it forwards the UIs

# 2. load a dataset (in a second terminal)
uv sync --all-packages
uv run python examples/datasets/prepare.py --dataset covertype --rows 20000 --upload

# 3. check, submit and follow a model
uv run modelport validate examples/models/covertype-classifier
uv run modelport submit examples/models/covertype-classifier --wait
uv run modelport status covertype-classifier
```

A model is three things in a folder:

1. **A `BaseModel` subclass** with `load`, `preprocess`, `predict` and `train_loop` ([SDK guide](docs/components/sdk.md)).
2. **A `model.yaml`** with its name, labels, input columns, training settings and gate thresholds ([reference](docs/components/model-yaml.md)).
3. **`example/input.parquet`**: a few input rows `modelport validate` runs `predict` on.

`modelport submit` does the rest. Nothing in the folder mentions Kubernetes, Ray clusters or Airflow.

## What happens after `submit`

```text
modelport submit DIR
  └─ validate, package code.zip, log it to MLflow
     └─ Airflow train_model
          └─ RayJob on its own Ray cluster: Ray Train on N workers, a checkpoint per epoch, resume on failure
               └─ register an MLflow version: weights, signature, lineage tags, drift baseline
                    └─ Airflow evaluate_and_promote (shared Ray cluster)
                         ├─ passed → state candidate, alias candidate
                         └─ failed → state rejected, every reason recorded
new file in s3://incoming/<dataset>/
  └─ Airflow ingest_dataset: quality checks → dataset v<N+1> or quarantine
       └─ retrain_on_dataset → train_model for every subscribed model
```

| Alias | Points at |
| --- | --- |
| `candidate` | The newest version that passed the gate. |
| `champion` | The version that serves traffic (from v0.2). |
| `previous` | The champion before the current one, for rollback (from v0.2). |

Details: [ingest](docs/components/ingest.md), [training](docs/components/training.md), [gate](docs/components/gate.md).

## Commands

| `modelport` | What it does |
| --- | --- |
| `modelport validate DIR` | Check a model folder locally, with no cluster. |
| `modelport submit DIR` | Validate, package and train; `--wait` blocks until the gate decides. |
| `modelport status NAME` | Aliases, the newest version's state and gate report, Airflow runs, links. |

All options and exit codes: [CLI reference](docs/components/cli.md).

| `make` | What it does |
| --- | --- |
| `make up` | Create the kind cluster and install everything (`PROFILE=full` or `smoke`). |
| `make down` | Delete the cluster and local state. |
| `make ui` | Port-forward Airflow, MLflow, MinIO, the Ray dashboard, Serve and Prometheus, and print the URLs and login. |
| `make images` | Build the three modelport images and load them into kind. |
| `make test` | Unit, contract and local integration tests (no cluster). |
| `make lint` | `ruff check` and `ruff format --check`. |
| `make typecheck` | `mypy --strict`. |
| `make help` | List the targets. |
| `make verify` | Check runs from their artifacts (not wired yet: run `uv run python scripts/verify.py --profile full --model NAME`). |
| `make demo` | Onboard both example models end to end (v0.3). |
| `make bench` | Measurements and failure drills for the results table (v0.2). |
| `make metrics-check` | Every Prometheus target up, every catalogue metric with data (v0.3). |
| `make rules-test` | Unit tests for the alert rules (v0.3). |

## Architecture

```text
kind cluster "modelport"
├─ platform    Airflow 3.4 (scheduler, API server, DAG processor, triggerer) · MLflow 3.18 · PostgreSQL 18 · MinIO (S3)
├─ ray         KubeRay operator · one RayJob cluster per training run · RayCluster modelport-shared
└─ monitoring  Prometheus and Grafana (from v0.2)
```

- **Airflow orchestrates, Ray computes.** Airflow is the only writer of aliases; Ray does the work.
- **Three Ray patterns:** a RayJob with its own cluster per training run (isolation, deleted when done), jobs on the shared autoscaling `modelport-shared` cluster for short evaluation and batch work, and, from v0.2, a RayService for online serving.
- **MLflow is the registry and the source of truth** for versions, lineage tags, aliases and gate reports; MinIO holds datasets, checkpoints and artifacts.

How it is installed, key by key: [infrastructure](docs/components/infrastructure.md). Architecture decisions: [docs/adr/](docs/adr/README.md). When something goes wrong: [runbook](docs/runbook.md).

## Scope of v0.1

Working: ingest with quality checks and quarantine, dataset versions, retraining on new data, distributed training with checkpoints and resume, registration with lineage and a drift baseline, the evaluation gate, and `validate`, `submit` and `status`. CI brings the whole platform up on kind and runs this flow end to end, including a deliberately broken model that must be rejected and a training worker killed mid-run.

Not yet: serving, canary and `promote` / `rollback` / `predict` (v0.2), batch inference (v0.2), drift detection, dashboards and alerts (v0.3). Measured results will be published from `make bench` in v0.2.

## Results

RayJob per job against the shared cluster, measured by `make bench` ([method](bench/README.md)):

<!-- bench:start -->
| Pattern | n | Startup s | Duration s | CPU-seconds |
| --- | --- | --- | --- | --- |
| RayJob (cluster per job) | 5 | 27.6 (27.6-28.6) | 11.8 (11.8-15.0) | 51.2 (48.7-53.4) |
| Shared cluster, workers from zero | 5 | 1.0 (1.0-3.0) | 27.3 (22.2-29.3) | 48.8 (43.8-57.7) |
| Shared cluster, worker already up | 5 | 1.0 (1.0-1.0) | 8.1 (8.1-9.1) | 11.7 (4.0-22.8) |

Median (min-max) of n runs per pattern. Host: runner: 4 vCPU, 15 GiB. Raw data: [bench-20261128T082637Z.csv](docs/results/raw/bench-20261128T082637Z.csv).
<!-- bench:end -->

## Known gaps

modelport is a development environment, not a production platform:

- One tenant, no SSO, no high availability; one kind node.
- Development credentials only: generated once by `make up`, stored in Kubernetes Secrets and `.modelport/credentials.env`.
- MinIO is built from its last community release, whose repository is archived (ADR-0003).
- CPU only; no GPU in CI.

## Versions

Every pinned version (Kubernetes, Ray, KubeRay, Airflow, MLflow, PostgreSQL, Python packages) is in [`versions.env`](versions.env) and the lock files.

## License and data credits

Apache-2.0 (see [LICENSE](LICENSE)). The example datasets are Fashion-MNIST (MIT), EuroSAT (MIT) and Covertype (CC BY 4.0); credits in [NOTICE](NOTICE) and [examples/datasets/README.md](examples/datasets/README.md). MinIO (AGPL-3.0) runs unmodified as a separate service and is not distributed.
