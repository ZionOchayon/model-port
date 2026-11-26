"""`model.yaml`: the typed `ModelSpec`, its loader, and its exported JSON Schema.

The normative schema is the contract; `ModelSpec` accepts exactly the documents it accepts. Validation runs in
pydantic's strict JSON mode, so YAML values are checked the way JSON Schema checks them (no "2" for an
integer, no `true` for a number), and the cross-field rules JSON Schema cannot express are model validators.
"""

import hashlib
import json
import re
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any, Final, Literal, Self

import yaml
from pydantic import BaseModel as PydanticModel
from pydantic import ConfigDict, Field, StringConstraints, ValidationError, model_validator
from pydantic_core import ErrorDetails

from modelport.errors import SpecError

SCHEMA_ID: Final = "https://modelport.dev/schemas/model.schema.json"
PIP_PIN_PATTERN: Final = r"^[A-Za-z0-9][A-Za-z0-9_.-]*(\[[A-Za-z0-9_,.-]+\])?==[A-Za-z0-9_.+!-]+$"
DATASET_URI_PATTERN: Final = r"^s3://datasets/[a-z0-9][a-z0-9-]{0,38}[a-z0-9](/v[1-9][0-9]*)?/?$"
IDENTIFIER: Final = r"^[A-Za-z_][A-Za-z0-9_]*$"
_STRICT: Final = ConfigDict(extra="forbid", frozen=True, strict=True)

Quantity = Annotated[str, StringConstraints(pattern=r"^[1-9][0-9]*(Mi|Gi)$")]
PipPin = Annotated[str, StringConstraints(pattern=PIP_PIN_PATTERN)]
Label = Annotated[str, StringConstraints(min_length=1, max_length=64)]
ParamValue = str | int | float | bool | list[Any] | None


class Framework(StrEnum):
    """Training framework, which picks the Ray Train strategy."""

    TORCH = "torch"
    XGBOOST = "xgboost"


class ColumnType(StrEnum):
    """Column types of the model signature (MLflow `ColSpec` types)."""

    BINARY = "binary"
    STRING = "string"
    DOUBLE = "double"
    FLOAT = "float"
    LONG = "long"
    INTEGER = "integer"
    BOOLEAN = "boolean"


class MetricName(StrEnum):
    """Primary metric of the evaluation gate."""

    F1_MACRO = "f1_macro"
    ACCURACY = "accuracy"


class ColumnSpec(PydanticModel):
    """One input column."""

    model_config = _STRICT
    name: Annotated[str, StringConstraints(pattern=IDENTIFIER)]
    type: ColumnType


class SignatureSpec(PydanticModel):
    """Input columns (outputs are fixed: prediction, confidence)."""

    model_config = _STRICT
    inputs: Annotated[list[ColumnSpec], Field(min_length=1)]


class RuntimeSpec(PydanticModel):
    """Per-model dependencies, each pinned with ==."""

    model_config = _STRICT
    pip: tuple[PipPin, ...] = ()


class TrainResources(PydanticModel):
    """Ray Train workers (at least 2: DDP on two workers)."""

    model_config = _STRICT
    workers: Annotated[int, Field(ge=2, le=8)] = 2
    cpus_per_worker: Annotated[float, Field(gt=0, le=4)] = 1.0
    memory_per_worker: Quantity = "2Gi"


class ServeResources(PydanticModel):
    """Serve replicas of one model version."""

    model_config = _STRICT
    replicas: Annotated[int, Field(ge=1, le=4)] = 1
    cpus: Annotated[float, Field(gt=0, le=2)] = 0.5
    memory: Quantity = "1Gi"


class BatchResources(PydanticModel):
    """Batch inference and evaluation jobs on the shared cluster."""

    model_config = _STRICT
    cpus_per_task: Annotated[float, Field(gt=0, le=4)] = 1.0
    batch_size: Annotated[int, Field(ge=1, le=8192)] = 256


class Resources(PydanticModel):
    """Compute for training, serving and batch."""

    model_config = _STRICT
    train: TrainResources = Field(default_factory=TrainResources)
    serve: ServeResources = Field(default_factory=ServeResources)
    batch: BatchResources = Field(default_factory=BatchResources)


class TrainSpec(PydanticModel):
    """Training settings passed to `train_loop` through `TrainConfig`."""

    model_config = _STRICT
    epochs: Annotated[int, Field(ge=1, le=200)] = 3
    batch_size: Annotated[int, Field(ge=1, le=8192)] = 64
    learning_rate: Annotated[float, Field(gt=0, le=10)] = 1e-3
    max_failures: Annotated[int, Field(ge=0, le=5)] = 2
    params: dict[str, ParamValue] = Field(default_factory=dict)


class EvaluationSpec(PydanticModel):
    """The three-condition evaluation gate."""

    model_config = _STRICT
    metric: MetricName = MetricName.F1_MACRO
    min_improvement: Annotated[float, Field(ge=-1, le=1)] = 0.0
    max_drop_per_class: Annotated[float, Field(ge=0, le=1)] = 0.05
    max_latency_increase: Annotated[float, Field(ge=0, le=10)] = 0.20
    min_absolute: Annotated[float, Field(ge=0, le=1)] = 0.5
    latency_batch_size: Annotated[int, Field(ge=1, le=4096)] = 256
    latency_repeats: Annotated[int, Field(ge=3, le=50)] = 5


class ServingSpec(PydanticModel):
    """Canary and serving thresholds."""

    model_config = _STRICT
    canary_weight: Annotated[float, Field(gt=0, le=0.5)] = 0.1
    observation_window_minutes: Annotated[int, Field(ge=1, le=120)] = 10
    min_canary_requests: Annotated[int, Field(ge=1)] = 50
    latency_p95_ms: Annotated[float, Field(gt=0)] = 500.0
    max_error_rate: Annotated[float, Field(ge=0, le=1)] = 0.02
    auto_deploy: bool = True


class DriftSpec(PydanticModel):
    """PSI thresholds and live-drift sensitivity."""

    model_config = _STRICT
    psi_warning: Annotated[float, Field(gt=0)] = 0.1
    psi_critical: Annotated[float, Field(gt=0)] = 0.25
    min_samples: Annotated[int, Field(ge=50)] = 500
    bins: Annotated[int, Field(ge=3, le=50)] = 10
    live_confidence_drop: Annotated[float, Field(gt=0, le=1)] = 0.15
    live_class_share_increase: Annotated[float, Field(gt=0, le=1)] = 0.30
    inference_log_sample_rate: Annotated[float, Field(gt=0, le=1)] = 1.0


class DataSpec(PydanticModel):
    """The dataset a model trains on: `s3://datasets/<name>` (latest version) or `.../v<n>` (pinned)."""

    model_config = _STRICT
    dataset: Annotated[str, StringConstraints(pattern=DATASET_URI_PATTERN)]
    label_column: Annotated[str, StringConstraints(pattern=IDENTIFIER)] = "label"
    auto_retrain: bool = True

    @property
    def dataset_name(self) -> str:
        """Dataset name, e.g. `fashion-mnist`."""
        return self.dataset.removeprefix("s3://datasets/").strip("/").split("/")[0]

    @property
    def pinned_version(self) -> int | None:
        """The version in the URI, or None to train on the latest version."""
        match = re.search(r"/v([1-9][0-9]*)/?$", self.dataset)
        return int(match.group(1)) if match else None


class ModelSpec(PydanticModel):
    """Everything the platform needs to know about one model (`model.yaml`)."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, populate_by_name=True)
    schema_version: Literal[1] = 1
    name: Annotated[str, StringConstraints(pattern=r"^[a-z][a-z0-9-]{1,38}[a-z0-9]$")]
    class_: Annotated[str, StringConstraints(pattern=r"^[A-Za-z_][A-Za-z0-9_.]*:[A-Za-z_][A-Za-z0-9_]*$")] = (
        Field(alias="class")
    )
    framework: Framework
    labels: Annotated[list[Label], Field(min_length=2, json_schema_extra={"uniqueItems": True})]
    signature: SignatureSpec
    runtime: RuntimeSpec = Field(default_factory=RuntimeSpec)
    resources: Resources = Field(default_factory=Resources)
    train: TrainSpec = Field(default_factory=TrainSpec)
    evaluation: EvaluationSpec = Field(default_factory=EvaluationSpec)
    serving: ServingSpec = Field(default_factory=ServingSpec)
    drift: DriftSpec = Field(default_factory=DriftSpec)
    data: DataSpec

    @model_validator(mode="after")
    def _cross_field_rules(self) -> Self:
        problems = []
        if len(set(self.labels)) != len(self.labels):
            problems.append("labels: every label must be unique")
        names = [c.name for c in self.signature.inputs]
        if len(set(names)) != len(names):
            problems.append("signature.inputs: column names must be unique")
        if self.data.label_column in names:
            problems.append(
                f"signature.inputs: '{self.data.label_column}' is data.label_column, not an input"
            )
        if self.drift.psi_critical <= self.drift.psi_warning:
            problems.append("drift.psi_critical: must be greater than drift.psi_warning")
        if problems:
            raise ValueError("; ".join(problems))
        return self


def _field_path(loc: tuple[int | str, ...]) -> str:
    path = ""
    for part in loc:
        path += f"[{part}]" if isinstance(part, int) else (f".{part}" if path else str(part))
    return path


def _describe(error: ErrorDetails) -> str:
    """One `<field path>: <message>` line in the data scientist's terms."""
    path = _field_path(error["loc"])
    if error["type"] == "string_pattern_mismatch" and path.startswith("runtime.pip["):
        return f'{path}: "{error["input"]}" must pin an exact version with == (e.g. torch==2.15.0)'
    if error["type"] == "extra_forbidden":
        return f"{path}: unknown field (check the spelling; see the model.yaml reference)"
    if error["type"] == "value_error":
        return str(error["msg"]).removeprefix("Value error, ")
    return f"{path}: {error['msg']}"


def load_spec(path: Path) -> ModelSpec:
    """Read and validate a model.yaml.

    Raises:
        SpecError: the file is missing, is not valid YAML, or breaks the schema (all problems listed).
    """
    if not path.is_file():
        raise SpecError(
            what=f"{path.name} not found in {path.parent}",
            fix="Create model.yaml next to your model class (see examples/models/image-classifier).",
        )
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.MarkedYAMLError as err:
        mark = err.problem_mark
        where = f" at line {mark.line + 1}, column {mark.column + 1}" if mark else ""
        raise SpecError(
            what=f"{path.name} is not valid YAML{where}: {err.problem}",
            fix="Fix the YAML syntax.",
        ) from err
    if not isinstance(raw, dict):
        raise SpecError(
            what=f"{path.name} must be a mapping of fields", fix="Start with `name:` and `class:`."
        )
    try:
        return ModelSpec.model_validate_json(json.dumps(raw))
    except TypeError as err:  # YAML values JSON cannot hold, e.g. an unquoted date
        raise SpecError(
            what=f"{path.name} has a value that is not plain data: {err}", fix="Quote it."
        ) from err
    except ValidationError as err:
        problems = [_describe(e) for e in err.errors(include_url=False)]
        raise SpecError(
            what=f"{path.name} is invalid:\n  " + "\n  ".join(problems),
            fix="Fix the fields listed above.",
            details={"errors": problems},
        ) from None


def spec_sha256(path: Path) -> str:
    """sha256 of the file with CRLF normalised to LF, so the hash is the same on every OS."""
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def export_json_schema() -> dict[str, Any]:
    """The JSON Schema (draft 2020-12) of model.yaml generated from `ModelSpec`."""
    schema = ModelSpec.model_json_schema(by_alias=True)
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": SCHEMA_ID,
        **schema,
        "title": "modelport model.yaml",
    }
