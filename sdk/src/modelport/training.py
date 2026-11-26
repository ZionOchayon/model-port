"""The small API a `train_loop` uses inside Ray Train: its configuration, data shards, reports and resume."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TrainConfig:
    """What `BaseModel.train_loop` receives: settings from model.yaml for this training run."""

    model_name: str
    labels: tuple[str, ...]
    label_column: str
    epochs: int
    batch_size: int
    learning_rate: float
    params: Mapping[str, Any]
    num_workers: int
