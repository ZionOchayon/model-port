# modelport

[![CI](https://github.com/ZionOchayon/model-port/actions/workflows/ci.yml/badge.svg)](https://github.com/ZionOchayon/model-port/actions/workflows/ci.yml) [![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

A data scientist puts a model into production with one command, on a local Kubernetes cluster.

modelport is an open-source ML platform on kind: distributed training on Ray, dataset versioning with quality checks, an evaluation gate, online serving with a canary and zero-downtime promotion, and daily batch inference, all driven by `modelport submit`. Drift monitoring, dashboards and alerts (v0.3) are next; see [Scope](#scope-of-v02).

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

# 4. once its canary has passed (about 10 minutes for a first version), ask it for predictions
uv run modelport predict covertype-classifier examples/models/covertype-classifier/example/input.parquet
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
                         │    └─ Airflow deploy_candidate: canary beside the champion on Ray Serve
                         │         ├─ passed → zero-downtime swap, alias champion (old one: previous)
                         │         └─ failed → rolled back, state rejected, every reason recorded
                         └─ failed → state rejected, every reason recorded
every day: Airflow batch_inference → each champion scores s3://batch-input/<dataset>/<day>/ on the shared cluster
new file in s3://incoming/<dataset>/
  └─ Airflow ingest_dataset: quality checks → dataset v<N+1> or quarantine
       └─ retrain_on_dataset → train_model for every subscribed model
```

| Alias | Points at |
| --- | --- |
| `candidate` | The newest version that passed the gate. |
| `champion` | The version that serves traffic and scores batch input. |
| `previous` | The champion before the current one, for rollback. |

Details: [ingest](docs/components/ingest.md), [training](docs/components/training.md), [gate](docs/components/gate.md), [serving](docs/components/serving.md), [canary and rollback](docs/components/canary.md), [batch](docs/components/batch.md).

## Commands

| `modelport` | What it does |
| --- | --- |
| `modelport validate DIR` | Check a model folder locally, with no cluster. |
| `modelport submit DIR` | Validate, package and train; `--wait` blocks until the gate decides. |
| `modelport status NAME` | Aliases, the newest version's state and gate report, Airflow runs, links. |
| `modelport predict NAME FILE` | Send rows from a JSON, CSV or Parquet file, or one image, to the champion. |
| `modelport promote NAME VERSION` | Make a version that passed the gate the champion, with zero downtime. |
| `modelport rollback NAME` | Make the previous champion the champion again. |

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
| `make bench` | Measure the Ray patterns and run the failure drills for the results table (`REPS=5`). |
| `make metrics-check` | Every Prometheus target up, every catalogue metric with data (v0.3). |
| `make rules-test` | Unit tests for the alert rules (v0.3). |

## Architecture

```text
kind cluster "modelport"
├─ platform    Airflow 3.4 (scheduler, API server, DAG processor, triggerer) · MLflow 3.18 · PostgreSQL 18 · MinIO (S3)
├─ ray         KubeRay operator · one RayJob cluster per training run · RayCluster modelport-shared · RayService modelport-serve
└─ monitoring  Prometheus (kube-prometheus-stack; Grafana and Alertmanager on the full profile)
```

- **Airflow orchestrates, Ray computes.** Airflow is the only writer of aliases; Ray does the work.
- **Three Ray patterns** (below): a RayJob per training run, jobs on the shared cluster for evaluation and batch, and a RayService for online serving.
- **MLflow is the registry and the source of truth** for versions, lineage tags, aliases and gate reports; MinIO holds datasets, checkpoints and artifacts.

How it is installed, key by key: [infrastructure](docs/components/infrastructure.md). Architecture decisions: [docs/adr/](docs/adr/README.md). When something goes wrong: [runbook](docs/runbook.md).

## Ray patterns

| Pattern | Used for | Why |
| --- | --- | --- |
| RayJob with its own cluster | Training | Minutes of work on several workers: isolated, sized per model, resumed from a checkpoint, deleted when done. |
| Jobs on the shared `modelport-shared` cluster | Evaluation, batch inference | Short jobs: no cluster to create, workers scale from zero and back. |
| RayService `modelport-serve` | Online serving | Long-running apps per model version, each with its own runtime environment, changed without downtime. |

The [Results](#results) table measures the first two on the same job. A shared cluster with a worker already up is the cheap case: no cluster to create and about a quarter of the CPU-seconds. From zero workers, the shared cluster and a RayJob spend about the same 20-30 s getting a worker ready, but a RayJob counts it as startup and the shared cluster as duration. That is why evaluation and batch share one cluster while training gets its own.

## Scope of v0.2

Working: ingest with quality checks and quarantine, dataset versions, retraining on new data, distributed training with checkpoints and resume, registration with lineage and a drift baseline, the evaluation gate, serving on Ray Serve with a canary judged on live errors and latency, zero-downtime promotion and rollback, daily batch inference with reruns that never duplicate, and `validate`, `submit`, `status`, `predict`, `promote` and `rollback`. CI brings the whole platform up on kind and runs this flow end to end, including a deliberately broken model that must be rejected, a training worker killed mid-run, two models served side by side and a batch day run twice.

Not yet (v0.3): drift detection, Grafana dashboards and alerts, `make demo` and `make metrics-check`.

## Results

RayJob per job against the shared cluster, measured by `make bench` ([method](docs/components/bench.md)):

<!-- bench:start -->
| Pattern | n | Startup s | Duration s | CPU-seconds |
| --- | --- | --- | --- | --- |
| RayJob (cluster per job) | 5 | 27.6 (27.6-28.6) | 11.8 (11.8-15.0) | 51.2 (48.7-53.4) |
| Shared cluster, workers from zero | 5 | 1.0 (1.0-3.0) | 27.3 (22.2-29.3) | 48.8 (43.8-57.7) |
| Shared cluster, worker already up | 5 | 1.0 (1.0-1.0) | 8.1 (8.1-9.1) | 11.7 (4.0-22.8) |

Median (min-max) of n runs per pattern. Host: runner: 4 vCPU, 15 GiB. Raw data: [bench-20261128T082637Z.csv](docs/results/raw/bench-20261128T082637Z.csv).
<!-- bench:end -->

Failure drills, run once per bench in the same workflow ([outcomes](docs/results/drills.md)): a training worker killed after its first checkpoint resumes instead of restarting; a promotion under constant load fails no request; a canary that fails half its requests is rolled back while the champion keeps serving.

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
