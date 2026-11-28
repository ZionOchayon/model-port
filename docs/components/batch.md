# Batch inference

Once a day, every model with a champion scores that day's input on the shared Ray cluster `modelport-shared` (FR-12). A rerun replaces the day's predictions and never duplicates them.

## DAG `batch_inference`

Scheduled `@daily` (no catch-up, one run at a time). It can be triggered by hand for a given day or models.

| Param         | Default                | Effect                                                              |
| ------------- | ---------------------- | ------------------------------------------------------------------- |
| `day`         | the run's logical date | The day to score, `YYYY-MM-DD`.                                     |
| `models`      | every registered model | Only these models.                                                  |
| `force_rerun` | `false`                | Run again the jobs that already succeeded for this day and version. |

```bash
uv run python scripts/ops.py trigger batch_inference manual_2026_11_28 --conf '{"day": "2026-11-28"}'
```

```text
plan → run_batch (one mapped task per model, pool shared_cluster) → summary
```

1. **`plan`** finds each model's champion and locks its version for the run. Models with no champion, or with no files under the day's input folder, are skipped and logged.
2. **`run_batch`** submits one Ray job per model to `modelport-shared` through the Jobs API and waits for it (up to an hour). It pushes the job's startup and duration to the Pushgateway (`pattern="shared"`, `kind="batch"`) and returns the rows the job wrote.
3. **`summary`** logs one line per model with its version and row count.

## Inputs and outputs

|        | Location                                                                                                                                                                 |
| ------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Input  | `s3://batch-input/<dataset>/<YYYY-MM-DD>/*.parquet`, where `<dataset>` is the champion's `data.dataset`. Every signature column must be there; other columns are kept.   |
| Output | `s3://predictions/<model>/<YYYY-MM-DD>/*.parquet`: the input columns plus `prediction`, `confidence`, `model` and `version`.                                             |
| Marker | `s3://predictions/<model>/<YYYY-MM-DD>/_SUCCESS.json`: `model`, `version`, `rows`, `input_uri`, `day`, `finished_at`. Written last; a folder without it is not finished. |

`uv run python examples/datasets/prepare.py --dataset covertype --batch-input --day YYYY-MM-DD --upload` uploads a day of input (no label column) for an example dataset.

The job reads the input with Ray Data and scores it in batches of `resources.batch.batch_size` rows with `resources.batch.cpus_per_task` CPUs per task, loading the version once per worker actor through the same MLflow model that serving loads. Its pip packages are the version's own, as in serving.

## Version locking

`plan` records the champion's version before any job starts, and each job loads that version by number, not by alias. A promotion while the run is in progress does not change what the run scores; the next run picks up the new champion. The marker names the version that produced the predictions.

## Reruns and idempotency

Each job has a fixed ID, `batch__<model>__<day>-v<version>`:

- A running job with that ID is adopted, not started twice (an Airflow retry waits for it).
- A succeeded job is skipped unless `force_rerun` is `true`.
- A failed job is deleted and submitted again; its log tail is written to the task log first.

The job overwrites its output folder as a whole, then writes `_SUCCESS.json`. Rerunning a day with `force_rerun` therefore replaces the predictions: check V-12 runs a day twice and requires the same rows with the same predictions, one per input row.

## When something goes wrong

- **The run succeeded but a model has no predictions:** it had no champion or no input that day; the `plan` task log says which (`batch.no_input`).
- **`run_batch` failed:** the task log has the job's log tail, and the Ray dashboard (`make ui`, `localhost:8265`) has the full job log under its ID. Rerun the task or the DAG run; a succeeded model is not run again.
