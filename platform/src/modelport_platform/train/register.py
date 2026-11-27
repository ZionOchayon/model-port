"""Register a trained model as a new MLflow version: pyfunc, signature, lineage tags, drift baseline."""

import dataclasses
import json
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import mlflow
import pandas as pd

from modelport.lifecycle import ModelState
from modelport.loader import instantiate
from modelport.logs import get_logger
from modelport.naming import Tag
from modelport.pyfunc import log_version
from modelport.registry import VersionInfo
from modelport.spec import Framework, ModelSpec
from modelport_platform.drift.baseline import build_baseline

_log = get_logger(__name__)


@dataclass(frozen=True)
class Lineage:
    """Where a version came from (spec section 5); every field becomes a version tag."""

    dataset: str
    dataset_version: int
    dataset_sha256: str
    val_uri: str
    code_sha256: str
    spec_sha256: str
    submission_run_id: str
    train_run_id: str
    train_rayjob: str
    image: str
    runtime_pip: tuple[str, ...]
    framework: Framework

    def to_tags(self) -> dict[Tag, str]:
        """Version tags (runtime_pip as a JSON list)."""
        return {
            Tag.DATASET: self.dataset,
            Tag.DATASET_VERSION: str(self.dataset_version),
            Tag.DATASET_SHA256: self.dataset_sha256,
            Tag.VAL_URI: self.val_uri,
            Tag.CODE_SHA256: self.code_sha256,
            Tag.SPEC_SHA256: self.spec_sha256,
            Tag.SUBMISSION_RUN_ID: self.submission_run_id,
            Tag.TRAIN_RUN_ID: self.train_run_id,
            Tag.TRAIN_RAYJOB: self.train_rayjob,
            Tag.IMAGE: self.image,
            Tag.RUNTIME_PIP: json.dumps(list(self.runtime_pip)),
            Tag.FRAMEWORK: str(self.framework),
        }


class VersionRegistry(Protocol):
    """The registry calls registration needs."""

    def find_version_by_tag(self, name: str, key: Tag, value: str) -> VersionInfo | None:
        """The version whose tag `key` equals `value`."""

    def set_version_tags(self, name: str, version: str, tags: Mapping[Tag, str]) -> None:
        """Set tags on a version."""

    def get_version(self, name: str, version: str) -> VersionInfo:
        """One version."""

    def log_json(self, run_id: str, artifact_path: str, payload: Mapping[str, Any]) -> None:
        """Write a JSON artifact to a run."""


def register_version(
    *,
    spec: ModelSpec,
    spec_path: Path,
    model_dir: Path,
    weights_dir: Path,
    val: pd.DataFrame,
    example: pd.DataFrame,
    lineage: Lineage,
    registry: VersionRegistry,
) -> VersionInfo:
    """Create the version once per training RayJob; a rerun returns the existing one.

    Uses the global MLflow tracking URI (the caller sets it) and reopens the training run.
    """
    existing = registry.find_version_by_tag(spec.name, Tag.TRAIN_RAYJOB, lineage.train_rayjob)
    if existing is not None:
        _log.info("register.exists", model=spec.name, version=existing.version, rayjob=lineage.train_rayjob)
        return existing
    model = instantiate(model_dir, spec)
    model.load(weights_dir)
    # built before the version exists, so a model that cannot predict on val never gets registered
    baseline = build_baseline(model, val, model_name=spec.name, version="", bins=spec.drift.bins)
    with mlflow.start_run(run_id=lineage.train_run_id), tempfile.TemporaryDirectory() as tmp:
        mlflow.log_artifact(str(spec_path), "spec")
        example_path = Path(tmp) / "input.parquet"
        example.to_parquet(example_path, index=False)
        mlflow.log_artifact(str(example_path), "example")
        mlflow.log_dict({"pip": list(lineage.runtime_pip)}, "train/runtime_pip.json")
        logged = log_version(
            spec=spec,
            spec_path=spec_path,
            model_dir=model_dir,
            weights_dir=weights_dir,
            example=example,
            registered_model_name=spec.name,
        )
    # shortcut: a crash between log_version and this call leaves an untagged version that a resubmitted
    # driver does not find (it creates another); look versions up by run_id if that is ever seen.
    registry.set_version_tags(
        spec.name, logged.version, {**lineage.to_tags(), Tag.STATE: ModelState.REGISTERED}
    )
    baseline = dataclasses.replace(baseline, version=logged.version)
    registry.log_json(lineage.train_run_id, "drift/baseline.json", baseline.to_json())
    _log.info(
        "register.version_created",
        model=spec.name,
        version=logged.version,
        dataset_version=lineage.dataset_version,
        code_sha256=lineage.code_sha256,
    )
    _log.info("register.baseline", features=len(baseline.features), rows=baseline.n_rows)
    return registry.get_version(spec.name, logged.version)
