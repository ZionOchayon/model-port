"""The model signature: input and output frames checked against `model.yaml` (both ends of the contract)."""

import math
from collections.abc import Callable
from typing import Final

import numpy as np
import pandas as pd
from mlflow.models import ModelSignature
from mlflow.types import ColSpec, Schema

from modelport.errors import SignatureMismatchError
from modelport.spec import ColumnSpec, ColumnType, ModelSpec

OUTPUT_COLUMNS: Final[tuple[tuple[str, ColumnType], ...]] = (
    ("prediction", ColumnType.STRING),
    ("confidence", ColumnType.DOUBLE),
)


def _is_real(v: object) -> bool:
    return isinstance(v, int | float | np.integer | np.floating) and not isinstance(v, bool | np.bool_)


def _is_int(v: object) -> bool:
    return isinstance(v, int | np.integer) and not isinstance(v, bool | np.bool_)


# type -> (accepts one value, what the value must be, how to fix it)
_RULES: Final[dict[ColumnType, tuple[Callable[[object], bool], str, str]]] = {
    ColumnType.BINARY: (
        lambda v: isinstance(v, bytes | bytearray),
        "bytes",
        "Send raw image bytes; the CLI base64-decodes JSON input for you.",
    ),
    ColumnType.STRING: (lambda v: isinstance(v, str), "text", "Send text values in this column."),
    ColumnType.DOUBLE: (
        lambda v: _is_real(v) and not math.isnan(float(v)),  # type: ignore[arg-type]
        "a number that is not NaN",
        "Fill or drop missing values; use type float in model.yaml if NaN is valid.",
    ),
    ColumnType.FLOAT: (_is_real, "a number", "Send numbers in this column."),
    ColumnType.LONG: (_is_int, "a whole number", "Send whole numbers (no decimals) in this column."),
    ColumnType.INTEGER: (_is_int, "a whole number", "Send whole numbers (no decimals) in this column."),
    ColumnType.BOOLEAN: (lambda v: isinstance(v, bool | np.bool_), "true or false", "Send booleans."),
}


def _first_bad_row(series: pd.Series, column: ColumnSpec) -> int | None:
    """Index position of the first incompatible value, or None. Numeric dtypes are checked vectorised."""
    kind = series.dtype.kind
    if column.type in (ColumnType.LONG, ColumnType.INTEGER) and kind in "iu":
        return None
    if column.type in (ColumnType.DOUBLE, ColumnType.FLOAT) and kind in "iuf":
        if column.type is ColumnType.FLOAT or kind in "iu":
            return None
        nan = np.flatnonzero(series.isna().to_numpy())
        return int(nan[0]) if len(nan) else None
    if column.type is ColumnType.BOOLEAN and kind == "b":
        return None
    accepts = _RULES[column.type][0]
    return next((i for i, v in enumerate(series.to_numpy()) if not accepts(v)), None)


def build_signature(spec: ModelSpec) -> ModelSignature:
    """The MLflow signature: model.yaml inputs; fixed outputs prediction (string), confidence (double)."""
    inputs = Schema([ColSpec(type=c.type.value, name=c.name) for c in spec.signature.inputs])
    outputs = Schema([ColSpec(type=t.value, name=name) for name, t in OUTPUT_COLUMNS])
    return ModelSignature(inputs=inputs, outputs=outputs)


def check_input_frame(frame: pd.DataFrame, spec: ModelSpec) -> None:
    """Every declared input column is present with compatible values. Extra columns are allowed.

    Raises:
        SignatureMismatchError: for the first missing column or incompatible value.
    """
    names = [c.name for c in spec.signature.inputs]
    for column in spec.signature.inputs:
        if column.name not in frame.columns:
            raise SignatureMismatchError(
                what=f"Input is missing column '{column.name}'",
                fix=f"Send the columns {', '.join(names)} (from signature.inputs in model.yaml).",
            )
        row = _first_bad_row(frame[column.name], column)
        if row is not None:
            _, must_be, fix = _RULES[column.type]
            value = frame[column.name].iloc[row]
            raise SignatureMismatchError(
                what=f"Column '{column.name}' must hold {must_be} (signature type {column.type}) "
                f"but row {row} holds {type(value).__name__}",
                fix=fix,
                details={"column": column.name, "row": row, "expected": column.type.value},
            )


def check_output_frame(output: pd.DataFrame, n_rows: int, spec: ModelSpec) -> None:
    """Exactly `prediction` and `confidence`, one row per input row, known labels, confidence in [0, 1].

    Raises:
        SignatureMismatchError: describing the first rule the output breaks.
    """
    expected = [name for name, _ in OUTPUT_COLUMNS]
    if list(output.columns) != expected:
        raise SignatureMismatchError(
            what=f"Output has columns {list(output.columns)}, expected {expected}",
            fix="Return modelport.Predictions(labels=[...], confidence=[...]) from predict().",
        )
    if len(output) != n_rows:
        raise SignatureMismatchError(
            what=f"predict returned {len(output)} predictions for {n_rows} input rows",
            fix="Return one label and one confidence per input row from predict().",
        )
    unknown = sorted(set(output["prediction"]) - set(spec.labels))
    if unknown:
        raise SignatureMismatchError(
            what=f"predict returned labels that are not in model.yaml: {unknown}",
            fix=f"Return only labels from `labels` in model.yaml: {list(spec.labels)}.",
        )
    confidence = output["confidence"].to_numpy(dtype=np.float64)
    bad = np.flatnonzero(~np.isfinite(confidence) | (confidence < 0) | (confidence > 1))
    if len(bad):
        raise SignatureMismatchError(
            what=f"confidence must be a number in [0, 1], but row {int(bad[0])} is {confidence[bad[0]]}",
            fix="Return probabilities (e.g. the softmax maximum) as confidence.",
        )
