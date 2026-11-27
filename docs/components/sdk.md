# modelport SDK

A model on modelport is a folder with three things: a `BaseModel` subclass, a [`model.yaml`](model-yaml.md), and `example/input.parquet` (a few input rows `modelport validate` runs `predict` on). `modelport submit` packages the folder; the platform trains, registers, evaluates and (from v0.2) serves it with the same class.

```python
import modelport
from modelport import BaseModel, Predictions, TrainConfig
```

Public API (`modelport.__all__`): `BaseModel`, `Predictions`, `TrainConfig`, `report`, `latest_artifacts`, `__version__`. The `modelport.training` module adds the helpers a `train_loop` needs.

## `BaseModel`

| Method                | Required | Called by                            | Contract                                                                                                                                                      |
| --------------------- | -------- | ------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `load(artifacts_dir)` | yes      | validate, evaluation, serving, batch | Load weights from `artifacts_dir` (the folder you reported as the checkpoint). `None` means "build an untrained model": `validate` and `train_loop` use that. |
| `preprocess(batch)`   | yes      | every inference                      | `batch` is a `pandas.DataFrame` with the `signature.inputs` columns. Return whatever `predict` takes.                                                         |
| `predict(inputs)`     | yes      | every inference                      | Return a `Predictions`: one label from `spec.labels` and one confidence in [0, 1] per row.                                                                    |
| `train_loop(config)`  | to train | Ray Train, once per worker           | The training loop; see below.                                                                                                                                 |
| `input_stats(batch)`  | no       | drift baseline                       | Per-row float features used to measure drift. Default: the numeric input columns. Image models return summary statistics (for example mean brightness).       |

`self.spec` is the parsed `model.yaml`. The platform calls `run_inference(batch)`, which checks the input against the signature, runs `preprocess` then `predict`, and checks the output: a wrong column, type, label or row count raises `SignatureMismatchError` (exit 4) with what is wrong. If `preprocess` or `predict` raises, or `predict` returns something other than `Predictions`, that is a `ContractError` (exit 3).

`Predictions(labels=[...], confidence=[...])` holds one label and one confidence per input row.

### A complete model (XGBoost)

From [`examples/models/covertype-classifier/model.py`](../../examples/models/covertype-classifier/model.py):

```python
class CovertypeClassifier(BaseModel):
    def load(self, artifacts_dir):
        self.labels = list(self.spec.labels)
        self.columns = [c.name for c in self.spec.signature.inputs]
        self.booster = None
        if artifacts_dir is not None:
            self.booster = xgboost.Booster()
            self.booster.load_model(str(Path(artifacts_dir) / "model.ubj"))

    def preprocess(self, batch):
        return xgboost.DMatrix(batch[self.columns].astype(np.float32))

    def predict(self, inputs):
        if self.booster is None:  # untrained: every class equally likely
            probs = np.full((inputs.num_row(), len(self.labels)), 1.0 / len(self.labels))
        else:
            probs = self.booster.predict(inputs)
        index = probs.argmax(axis=1)
        return Predictions(labels=[self.labels[i] for i in index], confidence=probs.max(axis=1))

    def train_loop(self, config: TrainConfig):
        self.load(None)
        frame = modelport.training.dataset_shard("train").materialize().to_pandas()
        index = {label: i for i, label in enumerate(config.labels)}
        dtrain = xgboost.DMatrix(
            frame[self.columns].astype(np.float32), label=frame[config.label_column].map(index).to_numpy()
        )
        previous, done = None, 0
        if (resume := modelport.latest_artifacts()) is not None:  # resume after a worker failure
            previous = xgboost.Booster()
            previous.load_model(str(resume / "model.ubj"))
            done = previous.num_boosted_rounds()
        params = {
            "objective": "multi:softprob",
            "num_class": len(config.labels),
            "tree_method": "hist",
            "eta": config.learning_rate,
            "max_depth": int(config.params.get("max_depth", 8)),
        }
        xgboost.train(
            params,
            dtrain,
            num_boost_round=max(config.epochs - done, 0),
            xgb_model=previous,
            callbacks=[modelport.training.xgboost_callback()],
        )  # a checkpoint every round
```

## Writing `train_loop`

`train_loop(config)` runs on every Ray Train worker (`resources.train.workers`, at least two). `config` is a `TrainConfig` built from `model.yaml`: `model_name`, `labels`, `label_column`, `epochs`, `batch_size`, `learning_rate`, `params` (your `train.params`) and `num_workers`.

The rules:

1. Read data only through `modelport.training.dataset_shard("train")` (or `"val"`): each worker gets its own shard of the dataset version being trained on.
2. Call `modelport.report(metrics, artifacts_dir=...)` once per epoch on every worker. `metrics` must contain `epoch` (an int, from 0). Pass `artifacts_dir` on rank 0 only: that folder becomes the checkpoint and, after the last epoch, the registered model's weights. `load()` must be able to read it.
3. Start with `modelport.latest_artifacts()`: it returns the last checkpoint as a local folder after a worker failure, or `None` on the first attempt. Continue from there instead of from epoch 0.

PyTorch with DDP, from [`examples/models/image-classifier/model.py`](../../examples/models/image-classifier/model.py):

```python
def train_loop(self, config: TrainConfig):
    from ray.train.torch import prepare_model

    self.load(None)
    start = 0
    if (resume := modelport.latest_artifacts()) is not None:
        self.net.load_state_dict(torch.load(resume / "model.pt", weights_only=True, map_location="cpu"))
        start = json.loads((resume / "meta.json").read_text())["epoch"] + 1
    net = prepare_model(self.net)  # DDP over gloo
    optimizer = torch.optim.Adam(net.parameters(), lr=config.learning_rate)
    shard = modelport.training.dataset_shard("train")
    for epoch in range(start, config.epochs):
        for batch in shard.iter_batches(batch_size=config.batch_size, batch_format="pandas"):
            ...  # forward, loss, backward, step
        metrics = {"epoch": epoch, "loss": loss_mean}
        if modelport.training.world_rank() == 0:
            with tempfile.TemporaryDirectory() as tmp:
                torch.save(self.net.state_dict(), Path(tmp) / "model.pt")
                (Path(tmp) / "meta.json").write_text(json.dumps({"epoch": epoch}))
                modelport.report(metrics, artifacts_dir=Path(tmp))
        else:
            modelport.report(metrics)
```

For XGBoost, pass `modelport.training.xgboost_callback()` to `xgboost.train`: it reports and checkpoints every boosting round and, like `report()`, logs the round's metrics and `checkpoint_saved` to MLflow on rank 0, so one round counts as one epoch.

| Helper                                             | Returns                                                                                                                                                                        |
| -------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `modelport.report(metrics, artifacts_dir=None)`    | Reports one epoch to Ray Train; on rank 0 also saves the checkpoint and logs the metrics to the MLflow training run (with `checkpoint_saved = 1` when a checkpoint was saved). |
| `modelport.latest_artifacts()`                     | The last checkpoint as a local `Path`, or `None`.                                                                                                                              |
| `modelport.training.dataset_shard(split)`          | This worker's `ray.data.DataIterator` of `"train"` or `"val"`.                                                                                                                 |
| `modelport.training.world_rank()` / `world_size()` | This worker's rank and the number of workers.                                                                                                                                  |
| `modelport.training.xgboost_callback()` | Ray's XGBoost report callback: a checkpoint every round, logged to MLflow like `report()`. |

## Signature types

`signature.inputs[].type` is one of `binary` (bytes, for example PNG images), `string`, `double`, `float`, `long`, `integer`, `boolean`. The outputs are always `prediction` (string, one of `labels`) and `confidence` (double).

## Errors

Every error says what failed and how to fix it, for example:

```text
Error: report() needs an 'epoch' metric (an int, starting at 0)
How to fix: Call modelport.report({'epoch': epoch, ...}) once per epoch.
```

The error classes and their CLI exit codes are listed in [cli.md](cli.md#exit-codes).
