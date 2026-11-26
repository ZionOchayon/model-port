"""`modelport validate`: check a model folder locally, with no cluster, and report each check with a fix.

Checks run in dependency order. After a failed structural check (folder, spec, class) the rest are
skipped, because they depend on it. Nothing here imports Ray or talks to the network.
"""

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

from modelport.base import BaseModel
from modelport.errors import ContractError, ModelportError
from modelport.loader import import_class, instantiate, require_concrete, require_subclass
from modelport.logs import get_logger
from modelport.signature import check_input_frame
from modelport.spec import ModelSpec, load_spec

EXAMPLE_FILE: Final = Path("example") / "input.parquet"
MAX_EXAMPLE_ROWS: Final = 1000
STATS_FIX: Final = "Return one row of float features per input row."
TRAIN_LOOP_FIX: Final = (
    "Implement train_loop(self, config) so the platform can train it; see docs/components/sdk.md#train_loop."
)
ORDER: Final = (
    "dir.exists",
    "spec.file",
    "spec.schema",
    "class.import",
    "class.subclass",
    "class.methods",
    "class.train_loop",
    "example.input",
    "model.load",
    "model.predict",
    "model.input_stats",
    "model.input_stats.coverage",
)
_log = get_logger(__name__)


class CheckStatus(StrEnum):
    """Outcome of one check."""

    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"
    SKIPPED = "skipped"


@dataclass(frozen=True)
class Check:
    """One named check (stable IDs such as `spec.schema`) and how to fix it if it failed."""

    name: str
    status: CheckStatus
    message: str
    fix: str | None = None


@dataclass(frozen=True)
class ValidationReport:
    """Every check of one model folder, in order."""

    model_dir: Path
    spec: ModelSpec | None
    checks: tuple[Check, ...]

    @property
    def ok(self) -> bool:
        """No check failed (warnings are allowed)."""
        return all(c.status is not CheckStatus.FAILED for c in self.checks)

    def first_failure(self) -> Check | None:
        """The first failed check, which is the one to fix first."""
        return next((c for c in self.checks if c.status is CheckStatus.FAILED), None)


class _Run:
    """Collects checks in order."""

    def __init__(self) -> None:
        self.checks: list[Check] = []

    def add(self, name: str, status: CheckStatus, message: str, fix: str | None = None) -> None:
        self.checks.append(Check(name, status, message, fix))

    def attempt[T](self, name: str, step: Callable[[], T], ok_message: str) -> tuple[bool, T | None]:
        """Run `step`; a ModelportError becomes a failed check carrying its fix."""
        try:
            value = step()
        except ModelportError as err:
            self.add(name, CheckStatus.FAILED, err.what, err.fix)
            return False, None
        self.add(name, CheckStatus.PASSED, ok_message)
        return True, value

    def skip_rest(self, reason: str) -> None:
        done = {c.name for c in self.checks}
        self.checks.extend(Check(n, CheckStatus.SKIPPED, reason) for n in ORDER if n not in done)


def validate_model_dir(
    model_dir: Path, *, spec_file: str = "model.yaml", run_predict: bool = True
) -> ValidationReport:
    """Check a model folder in dependency order and return every check.

    Args:
        model_dir: the folder with model.yaml and the model class.
        spec_file: spec file name inside the folder.
        run_predict: also load the model and run predict on example/input.parquet (checks 9-12).
    """
    run = _Run()
    spec = _structural_checks(model_dir, spec_file, run)
    if spec is not None:
        _model_checks(model_dir, spec, run, run_predict=run_predict)
    for check in run.checks:
        _log.debug("validate.check", check=check.name, status=check.status.value)
    return ValidationReport(model_dir=model_dir, spec=spec, checks=tuple(run.checks))


def _structural_checks(model_dir: Path, spec_file: str, run: _Run) -> ModelSpec | None:
    """Checks 1-7; returns the spec, or None after skipping the rest because a structural check failed."""
    if not model_dir.is_dir():
        run.add(
            "dir.exists",
            CheckStatus.FAILED,
            f"{model_dir} is not a folder",
            "Pass the folder that contains model.yaml.",
        )
        run.skip_rest("needs the model folder")
        return None
    run.add("dir.exists", CheckStatus.PASSED, str(model_dir))
    if not (model_dir / spec_file).is_file():
        run.add(
            "spec.file",
            CheckStatus.FAILED,
            f"{spec_file} not found in {model_dir}",
            "Create model.yaml; see examples/.",
        )
        run.skip_rest(f"needs {spec_file}")
        return None
    run.add("spec.file", CheckStatus.PASSED, spec_file)
    ok, spec = run.attempt("spec.schema", lambda: load_spec(model_dir / spec_file), f"{spec_file} is valid")
    if not ok or spec is None:
        run.skip_rest("needs a valid model.yaml")
        return None
    ok, found = run.attempt("class.import", lambda: import_class(model_dir, spec), spec.class_)
    if not ok or found is None:
        run.skip_rest("needs an importable model class")
        return None
    ok, cls = run.attempt("class.subclass", lambda: require_subclass(found), "inherits from BaseModel")
    if not ok or cls is None:
        run.skip_rest("needs a BaseModel subclass")
        return None
    ok, _ = run.attempt(
        "class.methods", lambda: require_concrete(cls), "load, preprocess and predict defined"
    )
    if not ok:
        run.skip_rest("needs load, preprocess and predict")
        return None
    if cls.implements_train_loop():
        run.add("class.train_loop", CheckStatus.PASSED, "train_loop implemented")
    else:
        run.add(
            "class.train_loop",
            CheckStatus.FAILED,
            f"{cls.__name__} does not implement train_loop",
            TRAIN_LOOP_FIX,
        )
    return spec


def _model_checks(model_dir: Path, spec: ModelSpec, run: _Run, *, run_predict: bool) -> None:
    """Checks 8-12: the example input, then load and predict on it."""
    example = _read_example(model_dir, spec, run)
    if not run_predict:
        run.skip_rest("skipped with --no-predict")
        return
    ok, model = run.attempt("model.load", lambda: _loaded(model_dir, spec), "load(None) works")
    if not ok or model is None or example is None:
        run.skip_rest("needs a loaded model and a valid example")
        return
    ok, _ = run.attempt(
        "model.predict", lambda: model.run_inference(example), "predict returned valid output"
    )
    if not ok:
        run.skip_rest("needs a working predict")
        return
    _check_input_stats(model, example, run)


def _read_example(model_dir: Path, spec: ModelSpec, run: _Run) -> pd.DataFrame | None:
    columns = ", ".join(c.name for c in spec.signature.inputs)
    fix = f"Add {EXAMPLE_FILE.as_posix()} with 1-{MAX_EXAMPLE_ROWS} rows and columns {columns}."
    path = model_dir / EXAMPLE_FILE
    if not path.is_file():
        run.add("example.input", CheckStatus.FAILED, f"{EXAMPLE_FILE.as_posix()} not found", fix)
        return None
    example = pd.read_parquet(path)
    if not 1 <= len(example) <= MAX_EXAMPLE_ROWS:
        run.add(
            "example.input", CheckStatus.FAILED, f"{EXAMPLE_FILE.as_posix()} has {len(example)} rows", fix
        )
        return None
    try:
        check_input_frame(example, spec)
    except ModelportError as err:
        run.add("example.input", CheckStatus.FAILED, err.what, err.fix)
        return None
    run.add("example.input", CheckStatus.PASSED, f"{len(example)} example rows match the signature")
    return example


def _loaded(model_dir: Path, spec: ModelSpec) -> BaseModel:
    model = instantiate(model_dir, spec)
    try:
        model.load(None)
    except ModelportError:
        raise
    except Exception as err:  # user code: report it with a fix instead of crashing validate
        raise ContractError(
            what=f"load(None) raised {type(err).__name__}: {err}",
            fix="load(None) must create an untrained model (no weights file yet).",
        ) from err
    return model


def _check_input_stats(model: BaseModel, example: pd.DataFrame, run: _Run) -> None:
    try:
        stats = model.input_stats(example)
    except Exception as err:  # user code: report it, never crash validate
        run.add(
            "model.input_stats",
            CheckStatus.FAILED,
            f"input_stats raised {type(err).__name__}: {err}",
            STATS_FIX,
        )
        return
    if not isinstance(stats, pd.DataFrame):
        problem: str | None = f"input_stats returned {type(stats).__name__}, not a DataFrame"
    elif len(stats) != len(example):
        problem = f"input_stats returned {len(stats)} rows for {len(example)} input rows"
    else:
        not_float = [str(c) for c, t in stats.dtypes.items() if not np.issubdtype(t, np.floating)]
        problem = f"input_stats columns are not float: {not_float}" if not_float else None
    if problem:
        run.add("model.input_stats", CheckStatus.FAILED, problem, STATS_FIX)
        return
    run.add("model.input_stats", CheckStatus.PASSED, f"{stats.shape[1]} drift features per row")
    if stats.shape[1] == 0:
        run.add(
            "model.input_stats.coverage",
            CheckStatus.WARNING,
            "input_stats returns no columns",
            "Drift will use only confidence and predicted class; override input_stats for input drift.",
        )
    else:
        run.add("model.input_stats.coverage", CheckStatus.PASSED, "input drift has features")
