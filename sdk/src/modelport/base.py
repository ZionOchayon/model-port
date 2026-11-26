"""`BaseModel`: the contract a data scientist implements (Template Method), and `Predictions`."""

import abc
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, final

import numpy as np
import pandas as pd

from modelport.errors import ContractError, ModelportError, SignatureMismatchError
from modelport.signature import check_input_frame, check_output_frame
from modelport.spec import ColumnType, ModelSpec
from modelport.training import TrainConfig

_NUMERIC = frozenset({ColumnType.DOUBLE, ColumnType.FLOAT, ColumnType.LONG, ColumnType.INTEGER})


@dataclass(frozen=True)
class Predictions:
    """One label and one confidence in [0, 1] per input row."""

    labels: Sequence[str]
    confidence: Sequence[float] | np.ndarray


class BaseModel(abc.ABC):
    """Subclass and implement load, preprocess, predict (and train_loop to train on the platform).

    The platform calls only `run_inference`, which fixes the order and checks both ends against the
    signature in model.yaml; subclasses override the hooks.
    """

    def __init__(self, spec: ModelSpec) -> None:
        self._spec = spec

    @property
    def spec(self) -> ModelSpec:
        """The parsed model.yaml."""
        return self._spec

    @abc.abstractmethod
    def load(self, artifacts_dir: Path | None) -> None:
        """Load weights from artifacts_dir; None means initialise an untrained model."""

    @abc.abstractmethod
    def preprocess(self, batch: pd.DataFrame) -> Any:  # noqa: ANN401 - model inputs are framework-specific
        """Turn raw input rows (columns from signature.inputs) into model inputs."""

    @abc.abstractmethod
    def predict(self, inputs: Any) -> Predictions:  # noqa: ANN401 - whatever preprocess returned
        """Return one label (from spec.labels) and one confidence per row."""

    def train_loop(self, config: TrainConfig) -> None:
        """Per-worker Ray Train loop. Call modelport.report() every epoch. Optional."""
        raise NotImplementedError

    def input_stats(self, batch: pd.DataFrame) -> pd.DataFrame:
        """Per-row float features for drift. Default: the numeric input columns."""
        columns = [c.name for c in self.spec.signature.inputs if c.type in _NUMERIC]
        return batch[columns].astype(np.float64)

    @final
    def run_inference(self, batch: pd.DataFrame) -> pd.DataFrame:
        """Run preprocess then predict, check both ends, and return DataFrame[prediction, confidence].

        Raises:
            SignatureMismatchError: the input or the output does not match the signature.
            ContractError: preprocess or predict raised, or predict returned something else.
        """
        check_input_frame(batch, self.spec)
        inputs = self._call_hook("preprocess", batch)
        predictions = self._call_hook("predict", inputs)
        if not isinstance(predictions, Predictions):
            raise ContractError(
                what=f"predict returned {type(predictions).__name__}, not Predictions",
                fix="Return modelport.Predictions(labels=[...], confidence=[...]).",
            )
        labels, confidence = list(predictions.labels), np.asarray(predictions.confidence, dtype=np.float64)
        if len(labels) != len(confidence):
            raise SignatureMismatchError(
                what=f"predict returned {len(labels)} labels but {len(confidence)} confidences",
                fix="Return one label and one confidence per input row from predict().",
            )
        output = pd.DataFrame({"prediction": labels, "confidence": confidence})
        check_output_frame(output, len(batch), self.spec)
        return output

    def _call_hook(self, name: str, arg: Any) -> Any:  # noqa: ANN401 - hooks are framework-specific
        try:
            return getattr(self, name)(arg)
        except ModelportError:
            raise
        except Exception as err:
            columns = ", ".join(c.name for c in self.spec.signature.inputs)
            fix = (
                f"Make preprocess accept a DataFrame with columns {columns}."
                if name == "preprocess"
                else "Make predict accept what preprocess returns and return modelport.Predictions."
            )
            raise ContractError(what=f"{name} raised {type(err).__name__}: {err}", fix=fix) from err

    @final
    @classmethod
    def implements_train_loop(cls) -> bool:
        """True when the subclass overrides train_loop, so the platform can train it."""
        return cls.train_loop is not BaseModel.train_loop
