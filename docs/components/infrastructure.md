# Infrastructure

Everything runs on one kind cluster named `modelport`, in three namespaces: `platform` (Airflow, MLflow, PostgreSQL, MinIO), `ray` (the KubeRay operator, training RayJobs, the shared RayCluster, the RayService `modelport-serve`) and `monitoring` (Prometheus, and on the full profile Grafana and Alertmanager). `make up` builds it from nothing; `make down` deletes it. Every pinned version is in [`versions.env`](../../versions.env).

## `make up`

`make up PROFILE=full|smoke` (default `full`) runs `scripts/up.sh`. Each step is idempotent, so running it again repairs a half-built stack. On failure it prints the pods that are not running and the latest warning events.

| Step           | What happens                                                                                                                                                                                                                                                    |
| -------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1 Preflight    | Docker reachable; engine memory at least 14 GiB (`full`) or 12 GiB (`smoke`); cgroup v2; a warning below 8 (or 4) CPUs.                                                                                                                                         |
| 2 Cluster      | `scripts/cluster.sh` creates the kind cluster (Kubernetes 1.37) if it does not exist.                                                                                                                                                                           |
| 3 Images       | `scripts/images.sh` builds `modelport/ray`, `modelport/airflow` and `modelport/minio`, tagged with the commit, and loads them into kind. Digest-pinned upstream images (PostgreSQL, MLflow) are re-tagged locally, because kind cannot load an image by digest. |
| 4 Namespaces   | `deploy/manifests/base/namespaces.yaml`.                                                                                                                                                                                                                        |
| 5 Secrets      | `scripts/secrets.sh` (below).                                                                                                                                                                                                                                   |
| 6 Monitoring | kube-prometheus-stack (`deploy/helm/kube-prometheus-stack.values.yaml`, plus `kube-prometheus-stack.smoke.values.yaml` on smoke: Prometheus and the exporters only). Its CRDs come first, so KubeRay can create its ServiceMonitor. Then the Pushgateway (`deploy/helm/pushgateway.values.yaml`), where Airflow tasks push job timings, gate and canary results and aliases. |
| 7 KubeRay      | The KubeRay operator chart with `deploy/helm/kuberay-operator.values.yaml`.                                                                                                                                                                                     |
| 8 Platform env | ConfigMap `modelport-platform-env` in `platform` and `ray`: service addresses, the Ray image, the profile and `MODELPORT_DATASETS`.                                                                                                                             |
| 9 Manifests    | `kubectl kustomize deploy/manifests/overlays/<profile>` with the image references filled in.                                                                                                                                                                    |
| 10 Storage     | Waits for PostgreSQL, MinIO, MLflow and the bucket Job.                                                                                                                                                                                                         |
| 11 Airflow     | The Apache Airflow chart with `deploy/helm/airflow.values.yaml` (plus `airflow.smoke.values.yaml` on smoke).                                                                                                                                                    |
| 12 Pools       | Airflow pools `ray_training` and `shared_cluster` (1 slot on smoke, 2 on full) and `serve_config` (1).                                                                                                                                                          |
| 13 Monitoring objects | `kubectl apply -k deploy/monitoring`: the PodMonitors that scrape every Ray head and worker in `ray`, the ServiceMonitors for MLflow, Airflow's statsd-exporter and MinIO, and the recording rules. |
| 14 Wait        | `scripts/wait_ready.py`: every pod Ready, the KubeRay CRDs present, `modelport-shared` ready, MLflow healthy, every DAG loaded with no import errors. Gives up after `UP_TIMEOUT` seconds (default 1500, exit 124).                                             |
| 15 Credentials | Writes `.modelport/credentials.env` (mode 600): the Airflow password for the CLI, and the S3 keys for platform tooling.                                                                                                                                         |

`make ui` port-forwards Airflow (`localhost:8080`), MLflow (`5000`), MinIO's S3 API (`9000`), the Ray dashboard (`8265`), Serve (`8000`) and Prometheus (`9090`), and prints the Airflow login.

## Profiles

|                        | `full`                             | `smoke` (CI)                     |
| ---------------------- | ---------------------------------- | -------------------------------- |
| Docker memory          | 14 GiB minimum, 16 GiB recommended | 12 GiB minimum                   |
| Parallel training jobs | 2                                  | 1                                |
| Shared cluster workers | up to 2                            | 1, smaller                       |
| Airflow                | parallelism 8                      | parallelism 4, smaller scheduler |

## Secrets

`scripts/secrets.sh` generates every credential once, with random values, and never overwrites an existing Secret. Nothing secret is stored in the repository, the images or the Helm values. These are development credentials for a local cluster, not a production secret store.

| Secret                                                                                   | Namespace         | Holds                                                                                    |
| ---------------------------------------------------------------------------------------- | ----------------- | ---------------------------------------------------------------------------------------- |
| `postgres-credentials`                                                                   | `platform`        | PostgreSQL superuser and the `airflow` and `mlflow` users.                               |
| `minio-credentials`                                                                      | `platform`        | MinIO root user.                                                                         |
| `modelport-s3`, `s3-credentials`                                                         | `platform`, `ray` | S3 keys as `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` for Airflow tasks and Ray pods. |
| `airflow-metadata`, `airflow-fernet-key`, `airflow-api-secret-key`, `airflow-jwt-secret` | `platform`        | Airflow's database connection and keys.                                                  |
| `airflow-auth`                                                                           | `platform`        | `passwords.json` for SimpleAuthManager (user `modelport`).                               |
| `airflow-connections`                                                                    | `platform`        | `AIRFLOW_CONN_MINIO_S3`, the S3 connection of the ingest sensor.                         |
| `grafana-admin`                                                                          | `monitoring`      | Grafana admin login (used from v0.2).                                                    |

## Images

| Image               | Built from                                                          | Contains                                                                                         |
| ------------------- | ------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------ |
| `modelport/ray`     | `deploy/images/ray` (hash-locked requirements)                      | Ray 2.60.1 with Train, Serve and Data, PyTorch (CPU), XGBoost, MLflow and the platform code.     |
| `modelport/airflow` | `deploy/images/airflow` (locked from the Airflow 3.4.0 constraints) | Airflow with the amazon and cncf providers, the platform code and the DAGs (baked in, ADR-0005). |
| `modelport/minio`   | `deploy/images/minio`                                               | MinIO built from source at its last community release (ADR-0003).                                |

## Manifests

Plain manifests for the stateful services (ADR-0004), in `deploy/manifests/`:

| File                                                 | What it creates                                                                                                                                                                            |
| ---------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `deploy/manifests/base/kustomization.yaml`           | The list of base manifests.                                                                                                                                                                |
| `deploy/manifests/base/namespaces.yaml`              | Namespaces `platform`, `ray`, `monitoring`.                                                                                                                                                |
| `deploy/manifests/base/postgres.yaml`                | PostgreSQL 18 StatefulSet (2 Gi volume) with an init script for the `airflow` and `mlflow` databases, and its Service.                                                                     |
| `deploy/manifests/base/minio.yaml`                   | MinIO Deployment with a 20 Gi volume and its Service (port 9000).                                                                                                                          |
| `deploy/manifests/base/buckets-job.yaml`             | Job that creates the eight buckets (`incoming`, `quarantine`, `datasets`, `predictions`, `mlflow`, `checkpoints`, `inference-logs`, `batch-input`) idempotently.                           |
| `deploy/manifests/base/mlflow.yaml`                  | MLflow 3.18 server (PostgreSQL backend, artifacts proxied to `s3://mlflow`, `--allowed-hosts` for the in-cluster names and localhost) and its Service.                                     |
| `deploy/manifests/base/airflow-rbac.yaml`            | Role and RoleBinding in `ray` for Airflow's scheduler and triggerer: manage RayJobs and RayServices, read RayClusters and pod logs.                                                        |
| `deploy/manifests/base/shared-raycluster.yaml`       | RayCluster `modelport-shared` for evaluation and batch jobs (ADR-0013): a head and a `batch` worker group that the in-tree autoscaler scales from 0 to 2, idle workers removed after 60 s. |
| `deploy/manifests/overlays/full/kustomization.yaml`  | The base, unchanged.                                                                                                                                                                       |
| `deploy/manifests/overlays/smoke/kustomization.yaml` | The base with smaller requests for PostgreSQL, MinIO and MLflow, and a shared cluster of at most one smaller worker.                                                                       |

## Helm values

### `deploy/helm/airflow.values.yaml`

Apache Airflow chart 1.22.0, Airflow 3.4.0 (ADR-0005).

| Key                                                                                                                   | Value and why                                                                                                    |
| --------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| `airflowVersion`                                                                                                      | `3.4.0`.                                                                                                         |
| `defaultAirflowRepository`, `defaultAirflowTag`                                                                       | `modelport/airflow`; the tag is set by `make up`.                                                                |
| `images.airflow.repository`, `images.airflow.tag`, `images.airflow.pullPolicy`                                        | The image built by `make up` (`IfNotPresent`: it is loaded into kind, never pulled).                             |
| `executor`                                                                                                            | `LocalExecutor`: tasks run in the scheduler pod; no Celery, no Redis.                                            |
| `enableServiceLinks`                                                                                                  | `false`: no Docker-link environment variables in pods.                                                           |
| `postgresql`, `pgbouncer`, `redis`                                                                                    | Disabled: PostgreSQL is the platform's own (`deploy/manifests/base/postgres.yaml`).                              |
| `data.metadataSecretName`                                                                                             | `airflow-metadata` (the database connection).                                                                    |
| `fernetKeySecretName`, `apiSecretKeySecretName`, `jwtSecretName`                                                      | The Secrets `scripts/secrets.sh` created.                                                                        |
| `createUserJob.enabled`                                                                                               | `false`: SimpleAuthManager reads its users from the config and passwords file.                                   |
| `migrateDatabaseJob.enabled`, `migrateDatabaseJob.useHelmHooks`, `migrateDatabaseJob.applyCustomEnv`                  | Run migrations as a plain Job (no Helm hook), so `helm --wait` works.                                            |
| `extraEnvFrom`                                                                                                        | ConfigMap `modelport-platform-env` and Secrets `modelport-s3` and `airflow-connections` in every Airflow pod.    |
| `config.core.parallelism`, `config.core.max_active_tasks_per_dag`                                                     | 8 and 8: what one scheduler pod can run.                                                                         |
| `config.core.load_examples`                                                                                           | `False`.                                                                                                         |
| `config.core.dags_are_paused_at_creation`                                                                             | `False`: DAGs ship in the image and must run without a manual unpause.                                           |
| `config.core.auth_manager`, `config.core.simple_auth_manager_users`, `config.core.simple_auth_manager_passwords_file` | SimpleAuthManager with one admin user `modelport`; the password file is copied from Secret `airflow-auth`.       |
| `config.dag_processor.refresh_interval`                                                                               | 60 s.                                                                                                            |
| `config.metrics.statsd_on`                                                                                            | `True`: task metrics to the StatsD exporter.                                                                     |
| `config.api.expose_config`                                                                                            | `False`.                                                                                                         |
| `config.logging.remote_logging`                                                                                       | `False`: task logs stay in the scheduler pod.                                                                    |
| `workers.celery.persistence`                                                                                          | Disabled (LocalExecutor).                                                                                        |
| `scheduler.replicas`, `scheduler.resources`, `scheduler.logGroomerSidecar.resources`                                  | One scheduler; it runs the tasks, so it gets the most memory.                                                    |
| `apiServer.replicas`, `apiServer.resources`                                                                           | One API server (UI and REST API).                                                                                |
| `apiServer.extraVolumes`, `apiServer.extraInitContainers`, `apiServer.extraVolumeMounts`                              | Copy `passwords.json` from Secret `airflow-auth` into a writable volume (SimpleAuthManager opens it for append). |
| `dagProcessor.enabled`, `dagProcessor.resources`                                                                      | The standalone DAG processor.                                                                                    |
| `triggerer.enabled`, `triggerer.persistence`, `triggerer.resources`                                                   | The triggerer runs deferred sensors (the ingest S3 sensor).                                                      |
| `statsd.enabled`, `statsd.resources`                                                                                  | StatsD exporter for Airflow metrics.                                                                             |
| `dags.persistence`, `dags.gitSync`                                                                                    | Disabled: DAGs are in the image.                                                                                 |
| `logs.persistence`                                                                                                    | Disabled.                                                                                                        |

### `deploy/helm/airflow.smoke.values.yaml`

Applied on top of `airflow.values.yaml` with `PROFILE=smoke`.

| Key                                                               | Value and why                |
| ----------------------------------------------------------------- | ---------------------------- |
| `config.core.parallelism`, `config.core.max_active_tasks_per_dag` | 4 and 4, for a CI runner.    |
| `scheduler.resources`, `apiServer.resources`                      | Smaller requests and limits. |

### `deploy/helm/kuberay-operator.values.yaml`

KubeRay operator chart 1.7.3.

| Key                                                 | Value and why                                                                                      |
| --------------------------------------------------- | -------------------------------------------------------------------------------------------------- |
| `image.repository`, `image.tag`, `image.pullPolicy` | `quay.io/kuberay/operator:v1.7.3`.                                                                 |
| `resources`                                         | Requests and limits of the operator pod.                                                           |
| `metrics.enabled`                                   | The operator exports metrics.                                                                      |
| `metrics.serviceMonitor` | Enabled: Prometheus scrapes the operator every 30 s with the operator's labels. |

### `deploy/helm/kube-prometheus-stack.values.yaml`

kube-prometheus-stack chart 93.11.1, installed as release `kps` in `monitoring` before KubeRay (step 6). Keys with a comment in the file are explained there; the others:

| Key | Value and why |
| --- | --- |
| `fullnameOverride` | `kps`: short object names (`kps-prometheus`, which `make ui` forwards to `localhost:9090`). |
| `kubeEtcd`, `kubeScheduler`, `kubeControllerManager`, `kubeProxy` | Disabled: kind does not expose these control-plane targets, so their scrapes would always be down. |
| `prometheus-node-exporter.resources`, `prometheusOperator.resources` | Small requests and limits for one node. |
| `prometheus.prometheusSpec.scrapeInterval` | `15s`: Serve and Ray metrics are read over canary windows of minutes. |
| `prometheus.prometheusSpec.evaluationInterval` | `30s` for recording and alert rules. |
| `prometheus.prometheusSpec.retentionSize` | `1GB`, with `retention` 2 days: a development stack. |
| `prometheus.prometheusSpec.serviceMonitorSelectorNilUsesHelmValues`, `prometheus.prometheusSpec.ruleSelectorNilUsesHelmValues`, `prometheus.prometheusSpec.probeSelectorNilUsesHelmValues`, `prometheus.prometheusSpec.scrapeConfigSelectorNilUsesHelmValues` | `false`: Prometheus selects every ServiceMonitor, rule, probe and scrape config in the cluster, not only the chart's own (the PodMonitors in `deploy/monitoring` and KubeRay's ServiceMonitor). |
| `prometheus.prometheusSpec.resources` | 512 Mi requested, 1.5 Gi limit. |
| `alertmanager.enabled`, `alertmanager.alertmanagerSpec.resources` | Alertmanager runs on the full profile, with small requests and limits. |
| `grafana.enabled`, `grafana.fullnameOverride`, `grafana.resources` | Grafana runs on the full profile as `grafana`, with small requests and limits. |

### `deploy/helm/kube-prometheus-stack.smoke.values.yaml`

Added on the smoke profile (CI): Prometheus and the exporters only.

| Key | Value and why |
| --- | --- |
| `alertmanager.enabled`, `grafana.enabled` | `false`: CI checks metrics in Prometheus; nothing reads alerts or dashboards there. |
| `prometheus.prometheusSpec.retention`, `prometheus.prometheusSpec.retentionSize` | 1 day and 512 MB: a CI run lasts under two hours. |
| `prometheus.prometheusSpec.resources` | 384 Mi requested, 1 Gi limit. |

### `deploy/helm/pushgateway.values.yaml`

prometheus-pushgateway chart 3.11.2, release `pushgateway` in `monitoring`. Every key is commented in the file. Airflow tasks push to `http://pushgateway.monitoring.svc.cluster.local:9091` (`MODELPORT_PUSHGATEWAY_URL` in the platform env); a failed push is logged and never fails a task.

### `deploy/helm/kube-prometheus-stack.slack.values.yaml`

Added by `make up` only when `SLACK_WEBHOOK_URL` is set in its environment; `scripts/secrets.sh` then stores the URL in the Secret `alertmanager-slack` in `monitoring`, and the values point Alertmanager at the mounted file (`api_url_file`), so the URL never appears in values or in the rendered config. Every alert but `Watchdog` goes to Slack, grouped by alert name, model and application, with resolved notices. Without it, alerts are visible in Alertmanager (`localhost:9093` on the full profile) and sent nowhere. CI renders both configs and checks them with `amtool check-config`.

## Monitoring objects (`deploy/monitoring/`)

| File | Objects |
| --- | --- |
| `deploy/monitoring/podmonitors.yaml` | PodMonitors `ray-head` and `ray-workers`: every Ray pod in `ray` on its `metrics` port every 15 s, labelled with its cluster (`ray_io_cluster`) and kind (`modelport_kind`). |
| `deploy/monitoring/servicemonitors.yaml` | ServiceMonitors `mlflow` (`/metrics`, from `--expose-prometheus`), `airflow-statsd` (the Airflow chart's statsd-exporter, port `statsd-scrape`) and `minio` (`/minio/v2/metrics/cluster`, public on this development stack). The `pushgateway` ServiceMonitor comes from its chart, with `honorLabels` so pushed labels are kept. |
| `deploy/monitoring/prometheus-rules.yaml` | PrometheusRule `modelport`, group `modelport.recording`: each champion's live confidence mean, prediction rate and class shares over 1 h next to its baseline and live-drift thresholds, and each Serve app's 5xx ratio and p95 latency over 5 min. `scripts/extract_rules.py` writes its rules to `deploy/monitoring/.rules.generated.yaml` for `promtool check rules`, which CI runs. |
| `deploy/monitoring/dashboards/*.json` | The four Grafana dashboards (`modelport-platform`, `modelport-serving`, `modelport-models`, `modelport-pipelines`), turned into ConfigMaps labelled `grafana_dashboard: "1"` by the kustomization; Grafana's sidecar loads them into the folder `modelport` (full profile). |
