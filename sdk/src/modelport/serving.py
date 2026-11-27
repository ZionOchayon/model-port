"""The HTTP serving contract shared by Serve and the CLI: request and response bodies, and input decoding.

A request holds one JSON object per row. Binary columns (images) travel as base64 strings; every other
value is a plain JSON value cast to the column's signature type. Type errors that survive the cast are
reported by `signature.check_input_frame`, so the messages are the same everywhere.
"""

import base64
import binascii
import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Annotated, Any, Final, Literal

import pandas as pd
from pydantic import BaseModel as PydanticModel
from pydantic import ConfigDict, Field, JsonValue, ValidationError

from modelport.errors import SignatureMismatchError
from modelport.spec import ColumnSpec, ColumnType

MAX_INSTANCES: Final = 1024
IMAGE_SUFFIXES: Final = (".png", ".jpg", ".jpeg")


class PredictRequest(PydanticModel):
    """`POST /<model>/predict` body: 1 to 1,024 rows."""

    model_config = ConfigDict(extra="forbid")
    instances: Annotated[list[dict[str, JsonValue]], Field(min_length=1, max_length=MAX_INSTANCES)]


class PredictionItem(PydanticModel):
    """One row's prediction."""

    prediction: str
    confidence: float


class PredictResponse(PydanticModel):
    """`POST /<model>/predict` answer: which version answered, and one prediction per row."""

    model: str
    version: str
    variant: Literal["champion", "candidate"]
    predictions: list[PredictionItem]


def _cast(series: pd.Series, kind: ColumnType) -> pd.Series:
    """Cast to the signature type when every value allows it; otherwise leave it for the signature check."""
    targets: Mapping[ColumnType, str] = {
        ColumnType.DOUBLE: "float64",
        ColumnType.FLOAT: "float64",
        ColumnType.LONG: "int64",
        ColumnType.INTEGER: "int64",
        ColumnType.BOOLEAN: "bool",
    }
    ok: Mapping[ColumnType, Callable[[object], bool]] = {
        ColumnType.DOUBLE: lambda v: isinstance(v, int | float) and not isinstance(v, bool),
        ColumnType.FLOAT: lambda v: isinstance(v, int | float) and not isinstance(v, bool),
        ColumnType.LONG: lambda v: isinstance(v, int) and not isinstance(v, bool),
        ColumnType.INTEGER: lambda v: isinstance(v, int) and not isinstance(v, bool),
        ColumnType.BOOLEAN: lambda v: isinstance(v, bool),
    }
    if kind in targets and series.dtype == object and all(ok[kind](v) for v in series):
        return pd.Series(series.tolist(), dtype=targets[kind])
    return series


def _decode_binary(values: Sequence[Any], column: str) -> list[Any]:
    decoded = []
    for row, value in enumerate(values):
        if not isinstance(value, str):
            decoded.append(value)  # the signature check reports the wrong type
            continue
        try:
            decoded.append(base64.b64decode(value, validate=True))
        except (binascii.Error, ValueError):
            raise SignatureMismatchError(
                what=f"row {row}, column '{column}': not valid base64",
                fix="Send binary columns (images) as base64 strings; `modelport predict` encodes files.",
            ) from None
    return decoded


def _select(frame: pd.DataFrame, spec_inputs: Sequence[ColumnSpec]) -> pd.DataFrame:
    missing = [c.name for c in spec_inputs if c.name not in frame.columns]
    if missing:
        raise SignatureMismatchError(
            what=f"input has no column {', '.join(repr(m) for m in missing)}",
            fix=f"Send every signature input: {', '.join(c.name for c in spec_inputs)}.",
        )
    return frame[[c.name for c in spec_inputs]].reset_index(drop=True)


def decode_request(req: PredictRequest, spec_inputs: Sequence[ColumnSpec]) -> pd.DataFrame:
    """The request's rows as a DataFrame in signature order (extra keys dropped, base64 decoded).

    Raises:
        SignatureMismatchError: a row lacks a signature column, or a binary value is not base64.
    """
    for row, instance in enumerate(req.instances):
        absent = [c.name for c in spec_inputs if c.name not in instance]
        if absent:
            raise SignatureMismatchError(
                what=f"row {row} has no column {', '.join(repr(a) for a in absent)}",
                fix=f"Send every signature input in every row: {', '.join(c.name for c in spec_inputs)}.",
            )
    columns: dict[str, pd.Series] = {}
    for column in spec_inputs:
        values = [instance[column.name] for instance in req.instances]
        if column.type is ColumnType.BINARY:
            columns[column.name] = pd.Series(_decode_binary(values, column.name), dtype=object)
        else:
            columns[column.name] = _cast(pd.Series(values, dtype=object), column.type)
    return pd.DataFrame(columns)


def _json_value(value: Any, kind: ColumnType) -> JsonValue:  # noqa: ANN401 - a DataFrame cell
    if kind is ColumnType.BINARY and isinstance(value, bytes | bytearray):
        return base64.b64encode(bytes(value)).decode()
    item = value.item() if hasattr(value, "item") else value  # numpy scalar -> Python
    return item  # type: ignore[no-any-return]


def encode_rows(frame: pd.DataFrame, spec_inputs: Sequence[ColumnSpec]) -> PredictRequest:
    """The request for these rows (binary columns as base64).

    Raises:
        SignatureMismatchError: a signature column is missing, or there are more than MAX_INSTANCES rows.
    """
    if len(frame) > MAX_INSTANCES:
        raise SignatureMismatchError(
            what=f"{len(frame)} rows is more than one request takes ({MAX_INSTANCES})",
            fix=f"Send at most {MAX_INSTANCES} rows, for example with `--rows {MAX_INSTANCES}`.",
        )
    selected = _select(frame, spec_inputs)
    kinds = {c.name: c.type for c in spec_inputs}
    rows = [{k: _json_value(v, kinds[str(k)]) for k, v in r.items()} for r in selected.to_dict("records")]
    return PredictRequest(instances=rows)


def read_input_file(path: Path, spec_inputs: Sequence[ColumnSpec]) -> pd.DataFrame:
    """Rows from a `.json`, `.csv` or `.parquet` file, or from one image file.

    A `.json` file is a request body or a list of rows; an image (`.png`, `.jpg`, `.jpeg`) works for a
    model with exactly one binary input column.

    Raises:
        SignatureMismatchError: unsupported file type, or the file does not fit the signature.
    """
    suffix = path.suffix.lower()
    if suffix in IMAGE_SUFFIXES:
        binary = [c.name for c in spec_inputs if c.type is ColumnType.BINARY]
        if len(spec_inputs) != 1 or len(binary) != 1:
            raise SignatureMismatchError(
                what=f"{path.name} is an image, but this model does not take exactly one binary input",
                fix="Pass a .json, .csv or .parquet file with every signature column.",
            )
        return pd.DataFrame({binary[0]: pd.Series([path.read_bytes()], dtype=object)})
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        try:
            request = (
                PredictRequest.model_validate(payload)
                if isinstance(payload, dict)
                else PredictRequest(instances=payload)
            )
        except ValidationError as err:
            raise SignatureMismatchError(
                what=f"{path.name} is not a request body or a list of rows ({err.error_count()} errors)",
                fix=f"Write 1 to {MAX_INSTANCES} JSON objects, one per row, keyed by the signature columns.",
            ) from None
        return decode_request(request, spec_inputs)
    if suffix == ".csv":
        strings = {c.name: str for c in spec_inputs if c.type is ColumnType.STRING}
        frame = _select(pd.read_csv(path, dtype=strings), spec_inputs)
        return frame.astype({c.name: "float64" for c in spec_inputs if c.type is ColumnType.DOUBLE})
    if suffix == ".parquet":
        return _select(pd.read_parquet(path), spec_inputs)
    raise SignatureMismatchError(
        what=f"cannot read input file type '{suffix or path.name}'",
        fix="Use .json, .csv, .parquet, or one .png/.jpg image.",
    )
