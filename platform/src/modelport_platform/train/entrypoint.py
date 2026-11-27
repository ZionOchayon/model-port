"""The training RayJob's entrypoint: `python -m modelport_platform.train.entrypoint --model … --rayjob …`.

The RayJob's working_dir is the submitted model folder, so the current directory holds model.yaml and the
model class. Exit codes: 0 trained, 1 training failed, 2 spec or data problem.
"""

import argparse
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from modelport.errors import ModelportError
from modelport.logs import configure_logging, get_logger
from modelport.naming import Tag, experiment_training
from modelport.spec import load_spec
from modelport_platform.ingest.versioning import DatasetVersion
from modelport_platform.ports import DatasetStore
from modelport_platform.settings import PlatformSettings
from modelport_platform.train.data import load_split, resolve_dataset_version
from modelport_platform.train.strategies import STRATEGIES, TrainContext, TrainerStrategy

_log = get_logger(__name__)


class TrainingRuns(Protocol):
    """The part of the registry the entrypoint needs."""

    def ensure_run(
        self, experiment: str, run_name: str, key_tag: Tag, tags: Mapping[str, str], params: Mapping[str, Any]
    ) -> str:
        """Find or create the training run."""


@dataclass
class Deps:
    """The entrypoint's adapters (fakes in tests)."""

    store: DatasetStore
    registry: TrainingRuns
    settings: PlatformSettings
    strategies: Mapping[str, TrainerStrategy] = field(default_factory=lambda: dict(STRATEGIES))
    read_split: Callable[[PlatformSettings, DatasetVersion, str], Any] = load_split
    cwd: Path = field(default_factory=Path.cwd)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """The RayJob entrypoint's arguments."""
    parser = argparse.ArgumentParser(prog="modelport_platform.train.entrypoint")
    parser.add_argument("--model", required=True)
    parser.add_argument("--submission-run-id", required=True)
    parser.add_argument("--spec-file", required=True)
    parser.add_argument("--rayjob", required=True)
    parser.add_argument("--dataset-version", type=int)
    return parser.parse_args(argv)


def _fail(code: int, what: str, fix: str) -> int:
    _log.error("train.failed", what=what, fix=fix)
    print(f"Error: {what}\nHow to fix: {fix}", file=sys.stderr)
    return code


def train(args: argparse.Namespace, deps: Deps) -> int:
    """Steps 1-7 of the training run; returns the exit code."""
    try:
        spec = load_spec(deps.cwd / args.spec_file)
    except ModelportError as err:
        return _fail(2, err.what, err.fix)
    if spec.name != args.model:
        return _fail(
            2, f"{args.spec_file} names '{spec.name}', not '{args.model}'", "Submit the model again."
        )
    try:
        version = resolve_dataset_version(deps.store, spec, args.dataset_version)
    except ModelportError as err:
        return _fail(2, err.what, err.fix)
    run_id = deps.registry.ensure_run(
        experiment_training(spec.name),
        args.rayjob,
        Tag.TRAIN_RAYJOB,
        tags={
            Tag.TRAIN_RAYJOB: args.rayjob,
            Tag.MODEL: spec.name,
            Tag.SUBMISSION_RUN_ID: args.submission_run_id,
            Tag.DATASET: version.dataset,
            Tag.DATASET_VERSION: str(version.version),
        },
        params={**spec.train.model_dump(exclude={"params"}), "workers": spec.resources.train.workers},
    )
    _log.info(
        "train.started",
        model=spec.name,
        rayjob=args.rayjob,
        dataset_version=version.version,
        workers=spec.resources.train.workers,
    )
    ctx = TrainContext(
        spec=spec,
        model_dir=deps.cwd,
        run_name=args.rayjob,
        mlflow_run_id=run_id,
        datasets={split: deps.read_split(deps.settings, version, split) for split in ("train", "val")},
        settings=deps.settings,
    )
    start = time.monotonic()
    try:  # Train V2 raises TrainingFailedError once max_failures restarts are used up
        result = deps.strategies[spec.framework].build_trainer(ctx).fit()
        error = result.error
    except Exception as err:  # any failure here is a failed training run (exit 1)
        _log.exception("train.fit_raised")
        error = err
    if error is not None:
        return _fail(1, f"Training failed: {error}", "Fix train_loop; the RayJob logs show the traceback.")
    if result.checkpoint is None:
        return _fail(
            1,
            "train_loop finished without reporting artifacts",
            "Call modelport.report(metrics, artifacts_dir) at least once from rank 0.",
        )
    _log.info(
        "train.finished",
        model=spec.name,
        seconds=round(time.monotonic() - start),
        metrics=dict(result.metrics),
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run inside the RayJob driver with the real adapters."""
    from modelport.registry import MlflowRegistry
    from modelport_platform.s3 import S3DatasetStore, boto3_client

    configure_logging(json=True)
    settings = PlatformSettings.from_env()
    deps = Deps(
        store=S3DatasetStore(boto3_client(settings)),
        registry=MlflowRegistry(settings.mlflow_uri),
        settings=settings,
    )
    return train(parse_args(argv), deps)


if __name__ == "__main__":
    sys.exit(main())
