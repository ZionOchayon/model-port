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

A URL that `make ui` printed does not answer: each port-forward writes to `/tmp/modelport-ui-<name>.log`, and `unable to listen on port ...: address already in use` there means another process holds 8080, 5000, 9000, 8265 or 8000, often an earlier `make ui`. Stop the old forwards (`pkill -f "kubectl.*port-forward"`) or the other process, then run `make ui` again.

## `modelport` says it cannot reach Airflow

`Cannot reach Airflow at http://localhost:8080` (exit 6): the port-forwards are not running. Run `make ui`. `No Airflow password found` (exit 11): run `make up`, which writes `.modelport/credentials.env`, from the repository folder.

## A version is `rejected`

That is the gate working, not a failure. `modelport status <model>` lists every reason with its numbers, and the report next to the champion's; see [components/gate.md](components/gate.md).
