# modelport (SDK and CLI)

The data scientist's side of [modelport](../README.md): the `BaseModel` contract your model implements, the helpers a `train_loop` uses on Ray Train, and the `modelport` CLI that validates and submits a model folder.

```bash
uv run modelport validate examples/models/covertype-classifier
uv run modelport submit examples/models/covertype-classifier --wait
uv run modelport status covertype-classifier
```

```python
import modelport
from modelport import BaseModel, Predictions, TrainConfig


class MyModel(BaseModel):
    # weights from a checkpoint folder, or None for an untrained model
    def load(self, artifacts_dir): ...

    # pandas DataFrame with the signature's columns -> model inputs
    def preprocess(self, batch): ...

    # -> Predictions(labels=[...], confidence=[...])
    def predict(self, inputs): ...

    # runs on every Ray Train worker; call modelport.report() once per epoch
    def train_loop(self, config: TrainConfig): ...
```

- [SDK guide](../docs/components/sdk.md): the contract, `train_loop` for PyTorch DDP and XGBoost, checkpoints and resume.
- [`model.yaml` reference](../docs/components/model-yaml.md): every field, its default and effect.
- [CLI reference](../docs/components/cli.md): commands, options, exit codes, environment.

Importing `modelport` and running `modelport validate` need no cluster and no Ray; Ray is imported only inside the training helpers. Python 3.12.
