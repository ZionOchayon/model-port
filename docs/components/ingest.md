# Ingest: dataset versions, quarantine and retraining

New training data arrives as Parquet files in `s3://incoming/<dataset>/`. The Airflow DAG `ingest_dataset` checks each file against the dataset's contract, then either writes the next immutable dataset version or moves the file to quarantine with a report. A new version triggers `retrain_on_dataset`, which retrains every model subscribed to that dataset.

```bash
uv run python examples/datasets/prepare.py --dataset fashion-mnist --rows 6000 --upload   # -> v1
uv run python examples/datasets/prepare.py --dataset fashion-mnist --corrupt --upload     # -> quarantine
```

## The contract

Each dataset has `s3://datasets/<dataset>/contract.json` (`examples/datasets/prepare.py` writes it on the first upload):

```json
{
  "dataset": "fashion-mnist",
  "label_column": "label",
  "columns": { "image": "binary", "label": "string" },
  "min_rows": 1000,
  "min_classes": 2,
  "max_class_share": 0.6,
  "image_format": "png"
}
```

| Key               | Default  | Meaning                                                                                                                                   |
| ----------------- | -------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| `dataset`         | required | The dataset name (lowercase, digits, `-`).                                                                                                |
| `label_column`    | required | The label column; it must be one of `columns`.                                                                                            |
| `columns`         | required | Every column and its type (the signature types: `binary`, `string`, `double`, `float`, `long`, `integer`, `boolean`), the label included. |
| `min_rows`        | `1000`   | Fewest rows a file may have.                                                                                                              |
| `min_classes`     | `2`      | Fewest distinct labels.                                                                                                                   |
| `max_class_share` | `0.6`    | Largest share one label may have.                                                                                                         |
| `image_format`    | none     | `png` or `jpeg`: every `binary` column must hold images of that format.                                                                   |

## Quality checks

Every file runs all checks, in this order, and the report lists each one with its detail:

| Check           | Passes when                                                                   |
| --------------- | ----------------------------------------------------------------------------- |
| `readable`      | The file is valid Parquet. An unreadable file fails every check.              |
| `schema`        | Every contract column is present with its type.                               |
| `rows`          | At least `min_rows` rows.                                                     |
| `nulls`         | No contract column has nulls.                                                 |
| `classes`       | At least `min_classes` labels.                                                |
| `class_balance` | No label above `max_class_share`.                                             |
| `images`        | Every value of the `binary` columns starts with the `image_format` signature. |

## Outcomes

| File                                                                               | Result                                                                                                                                                                                                                                                                                   |
| ---------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Passes every check                                                                 | `s3://datasets/<dataset>/v<N>/` with `train.parquet`, `val.parquet` (10 %, a split seeded by the file's sha256, so it is reproducible) and `manifest.json` (written last: a version without a manifest does not exist). An Airflow Asset event for `s3://datasets/<dataset>` is emitted. |
| Fails any check                                                                    | Copied to `s3://quarantine/<dataset>/<sha256>/` with `report.json` (every check, passed or failed, with its detail).                                                                                                                                                                     |
| Same bytes as an existing version                                                  | Nothing new: the existing version is kept and no event is emitted.                                                                                                                                                                                                                       |
| Dataset without `contract.json`, or a folder name that is not a valid dataset name | Quarantined with a single failed `schema` check, under `unknown-dataset` for an invalid name.                                                                                                                                                                                            |

In every case the file is removed from `incoming/`. A quarantined file is not an error: the task succeeds and logs `ingest.quarantined`.

`manifest.json` records `dataset`, `version`, `sha256`, `source_uri`, `rows_train`, `rows_val`, `labels`, `class_counts`, `created_at` and the quality report. Versions are never changed after they are written.

## DAG `ingest_dataset`

`schedule="@continuous"` with `max_active_runs=1`: one run at a time, and the next starts as soon as the previous ends. `start_date` is set: without it, Airflow 3.4.0 gives every continuous run after the first the same logical date and never creates them.

| Task            | What it does                                                                                                                                                 |
| --------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `wait_for_file` | A deferrable `S3KeySensor` on `s3://incoming/*/*.parquet` (connection `minio_s3`, poke every 30 s). It waits in the triggerer, so it holds no executor slot. |
| `list_files`    | Up to 10 incoming files, in name order.                                                                                                                      |
| `process`       | One mapped task per file: the checks, then version or quarantine. S3 errors are retried 3 times.                                                             |

## DAG `retrain_on_dataset`

Scheduled on the Asset of every dataset in `MODELPORT_DATASETS`. For each new version it finds the registered models tagged `modelport.dataset=<dataset>` and `modelport.auto_retrain=true` (`modelport submit` sets both from `data.dataset` and `data.auto_retrain`) and starts `train_model` for each with run ID `retrain__<model>__<dataset>__v<N>`. A retrain uses the model's latest submission (its code and spec file) on the new version. Starting the same retrain twice is a no-op.

To stop retraining a model, set `data.auto_retrain: false` and submit again.
