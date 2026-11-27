# `model.yaml` reference

`model.yaml` sits next to your model class and describes everything the platform needs: what the model is, what it reads, how to train it and how to judge it. `modelport validate` checks it against the JSON Schema in [`sdk/src/modelport/schema/model.schema.json`](../../sdk/src/modelport/schema/model.schema.json); editors that read JSON Schema can use the same file for completion.

A folder may hold several spec files (for example one per dataset); pick one with `--spec FILE`.

```yaml
schema_version: 1
name: covertype-classifier
class: model:CovertypeClassifier
framework: xgboost
labels:
  [
    Spruce/Fir,
    Lodgepole Pine,
    Ponderosa Pine,
    Cottonwood/Willow,
    Aspen,
    Douglas-fir,
    Krummholz,
  ]
signature:
  inputs:
    - { name: elevation, type: double }
    - { name: soil_type, type: long }
runtime:
  pip: [xgboost==3.4.1]
train: { epochs: 30, learning_rate: 0.3, params: { max_depth: 8 } }
evaluation: { metric: f1_macro, min_absolute: 0.5 }
data:
  dataset: s3://datasets/covertype
  label_column: cover_type
```

"Used from" says when the platform starts acting on a field; earlier releases accept and store it.

## Top level

| Field            | Type                 | Default        | Effect                                                                                                                                             |
| ---------------- | -------------------- | -------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| `schema_version` | integer              | `1`            | Version of this format. Only `1` exists.                                                                                                           |
| `name`           | string               | required       | Registered model name: lowercase letters, digits and `-`, 3 to 40 characters, starting with a letter. Every version, alias and run is keyed by it. |
| `class` | string | required | `module:Class` of your `BaseModel` subclass, relative to the model folder (`model:CovertypeClassifier` for `model.py`). |
| `framework`      | `torch` or `xgboost` | required       | Picks the Ray Train strategy: `TorchTrainer` with DDP over gloo, or `XGBoostTrainer`.                                                              |
| `labels`         | list of strings      | required       | The classes, at least two, unique. `predict` must return one of them per row.                                                                      |
| `signature`      | object               | required       | The model's input columns (outputs are always `prediction` and `confidence`).                                                                      |
| `runtime`        | object               | `{}`           | Extra dependencies for this model only.                                                                                                            |
| `resources`      | object               | defaults below | Compute for training, serving and batch.                                                                                                           |
| `train`          | object               | defaults below | Training settings passed to `train_loop` as a `TrainConfig`.                                                                                       |
| `evaluation`     | object               | defaults below | The evaluation gate.                                                                                                                               |
| `serving`        | object               | defaults below | Canary and serving thresholds (used from v0.2).                                                                                                    |
| `drift`          | object               | defaults below | Drift thresholds.                                                                                                                                  |
| `data`           | object               | required       | The dataset the model trains on.                                                                                                                   |

## `signature`

| Field              | Type                   | Default  | Effect                                                                                                                                                                                                        |
| ------------------ | ---------------------- | -------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `signature.inputs` | list of `{name, type}` | required | Input columns in order. `type` is one of `binary`, `string`, `double`, `float`, `long`, `integer`, `boolean`. Becomes the MLflow signature; inputs that do not match it are refused before `preprocess` runs. |

## `runtime`

| Field         | Type            | Default | Effect                                                                                                                                                                                                      |
| ------------- | --------------- | ------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `runtime.pip` | list of strings | `[]`    | Packages pinned with `==` (for example `xgboost==3.4.1`), installed for this model's training, evaluation and serving through Ray's `runtime_env`. Torch, Ray, pandas and pyarrow are already in the image. |

## `resources`

| Field                               | Type            | Default | Effect                                                                                   |
| ----------------------------------- | --------------- | ------- | ---------------------------------------------------------------------------------------- |
| `resources.train`                   | object          |         | Ray Train workers.                                                                       |
| `resources.train.workers`           | integer, 2 to 8 | `2`     | Number of training workers (each its own pod). At least two, so DDP is always exercised. |
| `resources.train.cpus_per_worker`   | number, up to 4 | `1.0`   | CPUs per training worker.                                                                |
| `resources.train.memory_per_worker` | quantity        | `2Gi`   | Memory limit of each worker pod.                                                         |
| `resources.serve`                   | object          |         | Serve replicas of one version (used from v0.2).                                          |
| `resources.serve.replicas`          | integer, 1 to 4 | `1`     | Replicas per version.                                                                    |
| `resources.serve.cpus`              | number, up to 2 | `0.5`   | CPUs per replica.                                                                        |
| `resources.serve.memory`            | quantity        | `1Gi`   | Memory per replica.                                                                      |
| `resources.batch`                   | object          |         | Jobs on the shared Ray cluster: evaluation now, batch inference from v0.2.               |
| `resources.batch.cpus_per_task`     | number, up to 4 | `1.0`   | CPUs of the evaluation (and batch) job.                                                  |
| `resources.batch.batch_size`        | integer         | `256`   | Rows per batch in batch inference (used from v0.2).                                      |

## `train`

| Field                 | Type              | Default | Effect                                                                                                                 |
| --------------------- | ----------------- | ------- | ---------------------------------------------------------------------------------------------------------------------- |
| `train.epochs`        | integer, 1 to 200 | `3`     | Epochs (for XGBoost: boosting rounds). `TrainConfig.epochs`.                                                           |
| `train.batch_size`    | integer           | `64`    | `TrainConfig.batch_size`.                                                                                              |
| `train.learning_rate` | number            | `0.001` | `TrainConfig.learning_rate`.                                                                                           |
| `train.max_failures`  | integer, 0 to 5   | `2`     | How many times Ray Train restarts the worker group after a worker dies; each restart resumes from the last checkpoint. |
| `train.params`        | mapping           | `{}`    | Anything else your `train_loop` reads, as `TrainConfig.params`.                                                        |

## `evaluation`

See [gate.md](gate.md) for how these are applied.

| Field                             | Type                     | Default    | Effect                                                                                    |
| --------------------------------- | ------------------------ | ---------- | ----------------------------------------------------------------------------------------- |
| `evaluation.metric`               | `f1_macro` or `accuracy` | `f1_macro` | Primary metric of the gate.                                                               |
| `evaluation.min_absolute`         | number, 0 to 1           | `0.5`      | The bar a model's first version must clear when there is no champion.                     |
| `evaluation.min_improvement`      | number, -1 to 1          | `0.0`      | How much the new version must beat the champion's primary metric by.                      |
| `evaluation.max_drop_per_class`   | number, 0 to 1           | `0.05`     | Largest allowed drop of any single class's score against the champion.                    |
| `evaluation.max_latency_increase` | number                   | `0.2`      | Largest allowed relative increase of inference latency against the champion (0.2 = 20 %). |
| `evaluation.latency_batch_size`   | integer                  | `256`      | Rows in the batch used to time inference.                                                 |
| `evaluation.latency_repeats`      | integer, 3 to 50         | `5`        | Timed repeats (after one warm-up); the median counts.                                     |

## `serving` (used from v0.2)

| Field                                | Type              | Default | Effect                                                     |
| ------------------------------------ | ----------------- | ------- | ---------------------------------------------------------- |
| `serving.auto_deploy`                | boolean           | `true`  | Start the canary as soon as a version passes the gate.     |
| `serving.canary_weight`              | number, up to 0.5 | `0.1`   | Share of live traffic sent to the candidate.               |
| `serving.observation_window_minutes` | integer           | `10`    | How long the canary is observed before it is judged.       |
| `serving.min_canary_requests`        | integer           | `50`    | Requests the candidate must serve before it can be judged. |
| `serving.latency_p95_ms`             | number            | `500.0` | Largest allowed p95 latency of the candidate.              |
| `serving.max_error_rate`             | number            | `0.02`  | Largest allowed error rate of the candidate.               |

## `drift`

| Field                             | Type                 | Default | Effect                                                                                   |
| --------------------------------- | -------------------- | ------- | ---------------------------------------------------------------------------------------- |
| `drift.bins`                      | integer, 3 to 50     | `10`    | Quantile bins of the drift baseline, built at registration from the validation split.    |
| `drift.psi_warning`               | number               | `0.1`   | PSI at which a feature counts as drifting (used from v0.3).                              |
| `drift.psi_critical`              | number               | `0.25`  | PSI at which drift is critical (used from v0.3).                                         |
| `drift.min_samples`               | integer, at least 50 | `500`   | Rows needed before PSI is computed (used from v0.3).                                     |
| `drift.live_confidence_drop`      | number               | `0.15`  | Drop of mean confidence that counts as live drift (used from v0.3).                      |
| `drift.live_class_share_increase` | number               | `0.3`   | Increase of one class's share of predictions that counts as live drift (used from v0.3). |
| `drift.inference_log_sample_rate` | number, up to 1      | `1.0`   | Share of served requests written to the inference log (used from v0.2).                  |

## `data`

| Field               | Type    | Default  | Effect                                                                                                                                         |
| ------------------- | ------- | -------- | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| `data.dataset`      | string  | required | `s3://datasets/<name>` for the latest version, or `s3://datasets/<name>/v<N>` to pin one. `modelport submit --dataset-version N` overrides it. |
| `data.label_column` | string  | `label`  | The label column in the dataset.                                                                                                               |
| `data.auto_retrain` | boolean | `true`   | Retrain this model whenever its dataset gets a new version (see [ingest.md](ingest.md)).                                                       |
