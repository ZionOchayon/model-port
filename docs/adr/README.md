# Architecture decision records

Each file records one decision: Context, Decision, Consequences.

| ADR | Decision |
| --- | --- |
| [ADR-0001](0001-record-architecture-decisions.md) | Record architecture decisions |
| [ADR-0002](0002-cli-name-modelport.md) | Name the CLI and SDK `modelport` (not `onboard`) |
| [ADR-0003](0003-minio-built-from-source.md) | Keep MinIO, built from source at its last community release |
| [ADR-0004](0004-plain-manifests-for-stateful-services.md) | Plain manifests for PostgreSQL, MinIO and MLflow; Helm only where charts are maintained |
| [ADR-0005](0005-airflow-3-localexecutor.md) | Airflow 3.4 on chart 1.22.0 with LocalExecutor, SimpleAuthManager and DAGs baked into the image |
| [ADR-0006](0006-rayjob-via-kubernetes-client.md) | Create RayJobs with the Kubernetes client and wait with a reschedule-mode sensor |
| [ADR-0007](0007-canary-via-serve-router.md) | Canary through a Serve-level router, not KubeRay incremental upgrade |
| [ADR-0008](0008-model-packaging.md) | Package code once: a zip for training, an MLflow pyfunc for everything after |
| [ADR-0009](0009-aliases-resolved-once.md) | Aliases are the only selector, resolved once per deployment or job |
| [ADR-0010](0010-airflow-single-writer.md) | Airflow is the only writer of aliases and of the RayService |
| [ADR-0011](0011-per-model-runtime-env.md) | Per-model dependencies through `runtime_env.pip` on an image with the frameworks preinstalled |
| [ADR-0012](0012-helm-3.md) | Use Helm v3.22.1, not Helm 4 |
| [ADR-0013](0013-shared-cluster-for-eval-and-batch.md) | Evaluation and batch jobs share one RayCluster through the Jobs API |
| [ADR-0014](0014-psi-method.md) | PSI with baseline quantile bins, epsilon 1e-4 and a 500-row minimum |
| [ADR-0015](0015-src-layout-uv-workspace.md) | Two Python distributions in a uv workspace with a `src` layout |
| [ADR-0016](0016-prometheus-in-v0-2.md) | Install Prometheus in v0.2, not v0.3 |
| [ADR-0017](0017-pushgateway-grouping.md) | Pushgateway groups keyed by model and source, replaced on every push |
| [ADR-0018](0018-serve-state-from-registry.md) | Render the whole RayService from registry state |
| [ADR-0019](0019-docker-dev-toolbox.md) | A Docker Compose toolbox for development tools; the platform stays on kind |
