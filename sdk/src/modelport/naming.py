"""The single source of every name modelport uses: aliases, tag keys, experiments, S3 paths and run IDs.

Every function validates its inputs and raises `SpecError` for a name that breaks the rules, so a bad name
fails where it is built instead of deep inside Airflow, Ray or MLflow.
"""

import hashlib
import re
from datetime import date
from enum import StrEnum
from typing import Final

from modelport.errors import SpecError

MODEL_NAME_PATTERN: Final = r"^[a-z][a-z0-9-]{1,38}[a-z0-9]$"  # <= 40 chars, DNS-safe
DATASET_NAME_PATTERN: Final = r"^[a-z0-9][a-z0-9-]{0,38}[a-z0-9]$"
DNS_1123: Final = re.compile(r"[a-z0-9]([-a-z0-9]{0,61}[a-z0-9])?")
_JOB_KEY = re.compile(r"[A-Za-z0-9_-]+")
_VERSION = re.compile(r"[1-9][0-9]*")
_SERVE_SYNC_ACTIONS: Final = frozenset({"promote", "rollback"})


class Alias(StrEnum):
    """The only three MLflow aliases modelport uses."""

    CANDIDATE = "candidate"
    CHAMPION = "champion"
    PREVIOUS = "previous"


class Tag(StrEnum):
    """MLflow tag keys on model versions, registered models and submission runs."""

    # model version tags
    STATE = "modelport.state"
    STATE_CHANGED_AT = "modelport.state_changed_at"
    FRAMEWORK = "modelport.framework"
    DATASET = "modelport.dataset"
    DATASET_VERSION = "modelport.dataset_version"
    DATASET_SHA256 = "modelport.dataset_sha256"
    VAL_URI = "modelport.val_uri"
    CODE_SHA256 = "modelport.code_sha256"
    SPEC_SHA256 = "modelport.spec_sha256"
    SUBMISSION_RUN_ID = "modelport.submission_run_id"
    TRAIN_RUN_ID = "modelport.train_run_id"
    TRAIN_RAYJOB = "modelport.train_rayjob"
    IMAGE = "modelport.image"
    RUNTIME_PIP = "modelport.runtime_pip"  # JSON list
    GATE_RESULT = "modelport.gate.result"  # "passed" | "failed"
    GATE_REASONS = "modelport.gate.reasons"  # JSON list of strings
    GATE_EVAL_RUN_ID = "modelport.gate.eval_run_id"
    CANARY_RESULT = "modelport.canary.result"  # "passed" | "failed"
    CANARY_REASONS = "modelport.canary.reasons"  # JSON list of strings
    # registered model tags
    LATEST_SUBMISSION_RUN_ID = "modelport.latest_submission_run_id"
    LATEST_CODE_SHA256 = "modelport.latest_code_sha256"
    LATEST_SPEC_FILE = "modelport.latest_spec_file"
    AUTO_RETRAIN = "modelport.auto_retrain"  # "true" | "false"
    # submission run tags
    MODEL = "modelport.model"
    SUBMISSION_KEY = "modelport.submission_key"


class Bucket(StrEnum):
    """S3 buckets created by `make up`."""

    INCOMING = "incoming"
    QUARANTINE = "quarantine"
    DATASETS = "datasets"
    PREDICTIONS = "predictions"
    MLFLOW = "mlflow"
    CHECKPOINTS = "checkpoints"
    INFERENCE_LOGS = "inference-logs"
    BATCH_INPUT = "batch-input"


EXPERIMENT_SUBMISSIONS: Final = "modelport-submissions"
EXPERIMENT_EVALUATION: Final = "modelport-evaluation"


def _model(name: str) -> str:
    if not re.fullmatch(MODEL_NAME_PATTERN, name):
        raise SpecError(
            what=f"'{name}' is not a valid model name",
            fix="Use 3-40 lowercase letters, digits and dashes, starting with a letter: image-classifier.",
            details={"name": name, "pattern": MODEL_NAME_PATTERN},
        )
    return name


def _dataset(name: str) -> str:
    if not re.fullmatch(DATASET_NAME_PATTERN, name):
        raise SpecError(
            what=f"'{name}' is not a valid dataset name",
            fix="Use 2-40 lowercase letters, digits and dashes (e.g. fashion-mnist).",
            details={"name": name, "pattern": DATASET_NAME_PATTERN},
        )
    return name


def _version_number(version: int) -> int:
    if version < 1:
        raise SpecError(what=f"Dataset version {version} is not valid", fix="Versions start at 1.")
    return version


def _version_string(version: str) -> str:
    if not _VERSION.fullmatch(version):
        raise SpecError(
            what=f"'{version}' is not a model version number", fix="Pass the number only, e.g. 7."
        )
    return version


def experiment_training(model: str) -> str:
    """MLflow experiment of a model's training runs: `modelport-train/<model>`."""
    return f"modelport-train/{_model(model)}"


def dataset_root(dataset: str) -> str:
    """`s3://datasets/<dataset>`."""
    return f"s3://{Bucket.DATASETS}/{_dataset(dataset)}"


def dataset_version_uri(dataset: str, version: int) -> str:
    """`s3://datasets/<dataset>/v<version>`."""
    return f"{dataset_root(dataset)}/v{_version_number(version)}"


def incoming_prefix(dataset: str) -> str:
    """`s3://incoming/<dataset>/`."""
    return f"s3://{Bucket.INCOMING}/{_dataset(dataset)}/"


def quarantine_uri(dataset: str, sha256: str) -> str:
    """`s3://quarantine/<dataset>/<sha256>/`."""
    return f"s3://{Bucket.QUARANTINE}/{_dataset(dataset)}/{sha256}/"


def checkpoint_root(model: str) -> str:
    """`s3://checkpoints/<model>` (Ray Train storage path)."""
    return f"s3://{Bucket.CHECKPOINTS}/{_model(model)}"


def prediction_uri(model: str, day: date) -> str:
    """`s3://predictions/<model>/<YYYY-MM-DD>/`."""
    return f"s3://{Bucket.PREDICTIONS}/{_model(model)}/{day.isoformat()}/"


def inference_log_prefix(model: str, day: date) -> str:
    """`s3://inference-logs/<model>/<YYYY-MM-DD>/`."""
    return f"s3://{Bucket.INFERENCE_LOGS}/{_model(model)}/{day.isoformat()}/"


def batch_input_uri(dataset: str, day: date) -> str:
    """`s3://batch-input/<dataset>/<YYYY-MM-DD>/`."""
    return f"s3://{Bucket.BATCH_INPUT}/{_dataset(dataset)}/{day.isoformat()}/"


def asset_uri(dataset: str) -> str:
    """The Airflow Asset of a dataset: `s3://datasets/<dataset>`."""
    return dataset_root(dataset)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def rayjob_name(model: str, run_key: str) -> str:
    """`train-<model>-<sha256(run_key)[:8]>`, at most 55 characters and DNS-1123 valid."""
    return f"train-{_model(model)}-{_sha256(run_key)[:8]}"


def submission_key(code_sha256: str, spec_file: str, spec_sha256: str) -> str:
    """Identity of one submission: the code, the spec file name and the spec content."""
    return _sha256(f"{code_sha256}:{spec_file}:{spec_sha256}")


def submit_run_id(model: str, submission_key: str) -> str:
    """`submit__<model>__<key[:12]>`: the train_model run started by `modelport submit`."""
    return f"submit__{_model(model)}__{submission_key[:12]}"


def retrain_run_id(model: str, dataset: str, version: int) -> str:
    """`retrain__<model>__<dataset>__v<n>`: the train_model run started by a new dataset version."""
    return f"retrain__{_model(model)}__{_dataset(dataset)}__v{_version_number(version)}"


def evaluate_run_id(model: str, version: str) -> str:
    """`evaluate__<model>__v<version>`."""
    return f"evaluate__{_model(model)}__v{_version_string(version)}"


def deploy_run_id(model: str, version: str) -> str:
    """`deploy__<model>__v<version>`."""
    return f"deploy__{_model(model)}__v{_version_string(version)}"


def serve_sync_run_id(action: str, model: str, stamp: str) -> str:
    """`<action>__<model>__<YYYYMMDDTHHMMSSZ>` for `promote` and `rollback`."""
    if action not in _SERVE_SYNC_ACTIONS:
        raise SpecError(what=f"'{action}' is not a serving action", fix="Use promote or rollback.")
    return f"{action}__{_model(model)}__{stamp}"


def serve_app_names(model: str) -> tuple[str, str, str]:
    """Serve application names: (router `<model>`, `<model>--champion`, `<model>--candidate`)."""
    m = _model(model)
    return m, f"{m}--champion", f"{m}--candidate"


def shared_job_id(kind: str, model: str, key: str) -> str:
    """`<kind>__<model>__<key>`: the Ray Jobs API submission ID on the shared cluster."""
    if not (_JOB_KEY.fullmatch(kind) and _JOB_KEY.fullmatch(key)):
        raise SpecError(
            what=f"'{kind}'/'{key}' cannot be part of a job ID",
            fix="Use letters, digits, '_' and '-' only.",
        )
    return f"{kind}__{_model(model)}__{key}"
