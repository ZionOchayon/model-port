# Training: RayJobs, checkpoints, resume and registration

Every submission trains as one KubeRay `RayJob` on its own Ray cluster, created by the Airflow DAG `train_model` and deleted when the job ends. The job runs your `train_loop` on Ray Train, checkpoints every epoch to S3, resumes from the last checkpoint after a failure, and registers the trained model in MLflow with its lineage and a drift baseline.

## DAG `train_model`

Started by `modelport submit` (run ID `submit__<model>__<code hash>`) or by `retrain_on_dataset` (`retrain__<model>__<dataset>__v<N>`). Conf: `model`, `submission_run_id`, `spec_file`, `dataset_version` (optional).

| Task                 | What it does                                                                                                                                                                                                                                                    |
| -------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `prepare`            | Downloads the submitted spec, resolves the dataset version (`--dataset-version`, then the version pinned in `data.dataset`, else the latest) and names the RayJob `train-<model>-<8 hex>` (a hash of the run ID, so a rerun of the same Airflow run reuses it). |
| `create_rayjob`      | Creates the RayJob through the Kubernetes API. Pool `ray_training` limits how many train at once (1 on smoke, 2 on full).                                                                                                                                       |
| `wait_rayjob`        | A sensor in reschedule mode (poke every 15 s, at most 2 h) that frees its slot between pokes. A failed RayJob fails the task with the command to read its logs: `kubectl -n ray logs job/<rayjob>`.                                                             |
| `record_version`     | Checks that the RayJob registered a version and returns its number.                                                                                                                                                                                             |
| `trigger_evaluation` | Starts `evaluate_and_promote` for that version (see [gate.md](gate.md)).                                                                                                                                                                                        |
| `push_job_timing`    | Pushes the RayJob's startup and run time to the Pushgateway.                                                                                                                                                                                                    |

## The RayJob

| Part       | Value                                                                                                                                                                                                                    |
| ---------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Cluster    | A head plus one worker group of `resources.train.workers` pods, each sized from `resources.train` (CPU, memory). Workers advertise one extra logical CPU so Ray Data can read while every training worker holds its CPU. |
| Image      | `modelport/ray`: Ray, PyTorch (CPU), XGBoost, pandas, pyarrow and the platform code.                                                                                                                                     |
| Code       | `runtime_env.working_dir` is the submission's `code.zip`, read from MLflow's artifact store; `runtime.pip` packages are added per model.                                                                                 |
| Entrypoint | `python -m modelport_platform.train.entrypoint --model <name> ...`                                                                                                                                                       |
| Lifetime   | `shutdownAfterJobFinishes: true` with a 60 s TTL, `activeDeadlineSeconds: 3600`, `backoffLimit: 1`, `submissionMode: K8sJobMode`.                                                                                        |

The entrypoint loads your class, reads the dataset version's `train` and `val` splits with Ray Data, and runs a `TorchTrainer` (DDP over gloo) or an `XGBoostTrainer` with your `train_loop` on every worker.

## Checkpoints and resume

`modelport.report(..., artifacts_dir=...)` on rank 0 saves a checkpoint per epoch to `s3://checkpoints/<model>/<rayjob>/` (the two newest are kept). When a worker dies, Ray Train restarts the worker group up to `train.max_failures` times; `modelport.latest_artifacts()` then returns the last checkpoint and `train_loop` continues from the next epoch instead of epoch 0. Rerunning a failed RayJob with the same name resumes from the same storage too.

The worker-failure drill checks this end to end: `bench/drills/kill_worker.sh` kills a Ray Train worker process after the first checkpoint, and `scripts/verify.py --checks V-05d` checks that the run succeeded and epoch 0 was trained only once.

## Registration and lineage

After training, the entrypoint registers the last checkpoint as a new MLflow model version:

- The MLflow pyfunc holds your code, the checkpoint folder, the spec and the signature built from `signature.inputs`, so evaluation and serving load it the same way.
- The version is tagged with its lineage: `modelport.dataset`, `modelport.dataset_version`, `modelport.dataset_sha256`, `modelport.val_uri`, `modelport.code_sha256`, `modelport.spec_sha256`, `modelport.submission_run_id`, `modelport.train_run_id`, `modelport.train_rayjob`, `modelport.image`, `modelport.runtime_pip`, `modelport.framework`, and its state `modelport.state=registered`.
- The training run gets `spec/model.yaml`, `train/runtime_pip.json` and `drift/baseline.json`: the quantile bins (`drift.bins`) of every `input_stats` column, `confidence` and `prediction` over up to 5,000 validation rows.
- Registration is idempotent: a RayJob that already registered a version does not register another.

Per-epoch metrics, including `checkpoint_saved`, are in the training run's metric history.
