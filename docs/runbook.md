# Runbook

Troubleshooting by symptom. Every command uses the cluster context `kind-modelport`; set it once with `k() { kubectl --context kind-modelport "$@"; }`.

## `make up` fails at preflight: engine memory

`preflight: engine memory 7 GiB < required 14 GiB` means Docker's engine has less memory than the profile needs. In Docker Desktop, raise the memory limit (Settings → Resources) to 16 GiB, or use `PROFILE=smoke` (12 GiB). On Windows with WSL 2, set `memory=16GB` under `[wsl2]` in `%UserProfile%\.wslconfig` and run `wsl --shutdown`.

## Pods stay `Pending`

```bash
k get pods -A --field-selector status.phase=Pending
k describe pod -n <ns> <pod> | tail -20
```

`Insufficient cpu` or `Insufficient memory` in the events means the node is full: wait for running training jobs to finish (`k get rayjobs -n ray`), lower `resources.train` in `model.yaml`, or give Docker more memory. A worker of `modelport-shared` that is Pending blocks evaluation; the cluster scales back to zero workers 60 s after its jobs end.

## A pod is `OOMKilled`

`k describe pod -n <ns> <pod>` shows `Last State: Terminated, Reason: OOMKilled`. For training workers, raise `resources.train.memory_per_worker` in `model.yaml` and submit again; Ray Train restarts a worker group up to `train.max_failures` times and resumes from the last checkpoint, so one kill is not fatal. For the Airflow scheduler (it runs every task with LocalExecutor), raise `scheduler.resources.limits.memory` in `deploy/helm/airflow.values.yaml` and run `make up` again.

## A RayJob is stuck or failed

```bash
k get rayjobs -n ray
k -n ray logs job/<rayjob>              # the submitter: the entrypoint's output and errors
k get pods -n ray -l modelport.dev/kind=train
```

- `jobStatus: FAILED`: the log ends with the Python error from your `train_loop` or the entrypoint. The `wait_rayjob` task's error message names the same command.
- `RUNNING` for a long time with no epochs in the MLflow run: check the worker pods are all Running; a worker that cannot be scheduled leaves Ray Train waiting for its workers.
- Pods are gone: a finished RayJob deletes its cluster 60 s after the end (`shutdownAfterJobFinishes`); its logs stay in Airflow and the Ray dashboard's job page.

## A new file in `s3://incoming/` is never versioned or quarantined

The `ingest_dataset` DAG should have a run waiting at `wait_for_file`:

```bash
sched=$(k get pod -n platform -l component=scheduler -o name)
k exec -n platform "${sched#pod/}" -c scheduler -- airflow dags list-runs ingest_dataset
k logs -n platform deploy/airflow-triggerer -c triggerer | tail -50
```

- No run in state `running`: look for `run already exists; skipping dagrun creation` in `k logs -n platform deploy/airflow-scheduler -c scheduler`. That means the continuous schedule is stuck on one logical date; the DAG must keep its `start_date`.
- The run waits but the file sits there: check the triggerer log for S3 errors from connection `minio_s3` (Secret `airflow-connections`).
- The file is gone but there is no version: it was quarantined; read `s3://quarantine/<dataset>/<sha256>/report.json`.

## DAG import errors

```bash
k exec -n platform deploy/airflow-scheduler -c scheduler -- airflow dags list-import-errors
```

DAGs are baked into the Airflow image, so a fix needs a new image: `make up` rebuilds and reloads it. `scripts/wait_ready.py` reports a missing DAG as `DAG <id> not loaded`.

## MLflow answers `403` or "Invalid Host header"

MLflow 3 checks the `Host` header against `--allowed-hosts`. The server allows its in-cluster names and `localhost`; reach it through `make ui` at `http://localhost:5000`, not through another hostname or IP. A new in-cluster name must be added to `--allowed-hosts` in `deploy/manifests/base/mlflow.yaml`.

## The MinIO image build fails

`scripts/images.sh` builds MinIO from source (ADR-0003), which downloads Go modules. A network failure during `go build` fails the build; rerun `make up` (a built image is reused). `make images` rebuilds the three images on their own.

## `make ui`: port already in use

A URL that `make ui` printed does not answer: each port-forward writes to `/tmp/modelport-ui-<name>.log`, and `unable to listen on port ...: address already in use` there means another process holds 8080, 5000, 9000, 8265, 8000 or 9090, often an earlier `make ui`. Stop the old forwards (`pkill -f "kubectl.*port-forward"`) or the other process, then run `make ui` again.

## `modelport` says it cannot reach Airflow

`Cannot reach Airflow at http://localhost:8080` (exit 6): the port-forwards are not running. Run `make ui`. `No Airflow password found` (exit 11): run `make up`, which writes `.modelport/credentials.env`, from the repository folder.

## A version is `rejected`

That is the gate working, not a failure. `modelport status <model>` lists every reason with its numbers, and the report next to the champion's; see [components/gate.md](components/gate.md).

## A canary was rolled back

That is the canary working: the version is `rejected`, the champion kept serving, and the `judge` task log and the version's `modelport.canary.reasons` tag in MLflow list every failed rule with its numbers ([components/canary.md](components/canary.md#the-rules)). `only N requests reached the candidate` means the window had too little traffic: on a local stack keep `synthetic_traffic` on, or trigger `deploy_candidate` again with a longer `observation_minutes`.

## A serving app does not start

`deploy_candidate` or `serve_sync` fails at a `wait_*` task with the app's message. `kubectl --context kind-modelport -n ray get rayservice modelport-serve -o yaml` shows each app's status. `signature self-test failed for <model> v<N>` means the version cannot score its own example input: fix the model and submit it again. An app stuck `DEPLOYING` usually waits for CPU or memory: see [Pods stay `Pending`](#pods-stay-pending), and lower `resources.serve` in `model.yaml`.

## `modelport promote` or `rollback` is refused

Exit 8 with the reason. `a canary is running for <model>`: wait for its `deploy_candidate` run to finish. `did not pass the gate`: `modelport status <model>` shows why. `no previous version to roll back to`: promote a version by number instead.

## `modelport predict` fails

`<model> has no champion` (exit 7): no version has finished a canary yet; `modelport status <model>` shows where the newest one is. `Cannot reach Serve at http://localhost:8000` (exit 6): run `make ui`, which forwards `localhost:8000`. A 422 names the column that does not fit the signature.

## A batch day has no predictions

See [components/batch.md](components/batch.md#when-something-goes-wrong): the model had no champion or no input under `s3://batch-input/<dataset>/<day>/`, or its job failed and the `run_batch` task log has the job's log tail.

## Alerts

Every alert links here by its name. Thresholds come from each model's `model.yaml` (`drift`, `serving`), except where a fixed value is given.

## `ModelDriftWarning`

The daily `drift_check` found a feature whose PSI against the champion's training baseline is at least `drift.psi_warning` (0.1 by default) and below `drift.psi_critical`. The live data is moving; nothing is broken. Look at the model's PSI per feature in Grafana (Models dashboard) and at recent input. If it keeps growing, retrain on newer data (a new dataset version retrains subscribed models). Drift never retrains a model by itself.

## `ModelDriftCritical`

Two ways to fire, told apart by the `layer` label:

- `layer="batch"`: a feature's PSI is at least `drift.psi_critical` (0.25 by default) in the daily check.
- `layer="live"`: for 15 minutes, with at least 100 predictions an hour, the champion's mean confidence over the last hour dropped more than `drift.live_confidence_drop` below its baseline, or a class's share grew more than `drift.live_class_share_increase`.

The model is seeing data unlike its training data, and its predictions may be wrong. Check the input source for a change (a new upstream, a broken preprocessing step), then retrain on data that includes the new distribution, or roll back with `modelport rollback NAME` if a recent promotion caused it. `simulate_drift` fires this alert on purpose.

## `ServeErrorRate`

More than 2% of the requests to a Serve app returned 5xx for 5 minutes. `kubectl --context kind-modelport -n ray get rayservice modelport-serve -o yaml` shows each app's status; the Ray dashboard (`make ui`, `localhost:8265`) has the replica logs, where each failure is logged as `router.variant_failed`. During a canary, the canary rolls a failing candidate back by itself; for the champion, roll back with `modelport rollback NAME`.

## `ServeLatencyP95`

The p95 latency of a Serve app has been above its model's `serving.latency_p95_ms` for 10 minutes. Usually the replicas are short of CPU: check node and pod CPU in Grafana, raise `resources.serve.replicas` or `resources.serve.cpus` in `model.yaml`, or lower the traffic.

## `CanaryFailed`

A candidate was rolled back in the last hour: the canary is working, and the champion kept serving. See [A canary was rolled back](#a-canary-was-rolled-back) for the reasons. The alert clears an hour after the canary's verdict.

## `DagFailed`

An Airflow DAG run failed after all its retries in the last 10 minutes. Open the run in Airflow (`make ui`, `localhost:8080`): the failed task's log has the error. A refusal (`AirflowFailException`, for example a promotion during a canary) is a deliberate failure with its reason in the log.

## `RayJobStuck`

A RayJob has been `Initializing` or `Waiting` for more than 10 minutes, almost always because its pods cannot be scheduled. See [Pods stay `Pending`](#pods-stay-pending) and [A RayJob is stuck or failed](#a-rayjob-is-stuck-or-failed).
