"""Package a trained BaseModel as one MLflow pyfunc (Adapter): Serve, batch and evaluation load it alike.

`log_version` stores the model folder as `code_paths`, the weights and the spec as artifacts, and the
signature from model.yaml. `load_context` finds the restored model folder, loads the class with the same
loader as `validate`, and calls `load(weights)`.
"""

import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlflow
import mlflow.pyfunc
import pandas as pd
from mlflow.exceptions import MlflowException
from mlflow.pyfunc.model import PythonModel, PythonModelContext

from modelport.base import BaseModel
from modelport.errors import NotFoundError
from modelport.loader import instantiate
from modelport.logs import get_logger
from modelport.signature import build_signature
from modelport.spec import ModelSpec, load_spec

ARTIFACT_PATH = "model"
_log = get_logger(__name__)


class ModelportPyfunc(PythonModel):
    """MLflow `PythonModel` around a modelport `BaseModel`."""

    _model: BaseModel

    def load_context(self, context: PythonModelContext) -> None:
        """Load the spec and the user's class from the restored model folder, then the weights."""
        spec_path = Path(context.artifacts["spec"])
        spec = load_spec(spec_path)
        code_dir = Path(context.artifacts["spec"]).parents[1] / "code" / context.model_config["code_dir"]
        model = instantiate(code_dir, spec)
        model.load(Path(context.artifacts["weights"]))
        self._model = model

    def predict(
        self,
        context: PythonModelContext | None,
        model_input: pd.DataFrame,
        params: dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        """Run the model's `run_inference` (input checked, two-column output)."""
        return self._model.run_inference(model_input)

    @property
    def model(self) -> BaseModel:
        """The loaded user model."""
        return self._model

    @property
    def spec(self) -> ModelSpec:
        """The model's spec."""
        return self._model.spec


@dataclass(frozen=True)
class LoggedVersion:
    """A registered model version created by `log_version`."""

    version: str
    run_id: str
    model_uri: str  # models:/<name>/<version>


def log_version(
    *,
    spec: ModelSpec,
    spec_path: Path,
    model_dir: Path,
    weights_dir: Path,
    example: pd.DataFrame,
    registered_model_name: str,
    extra_artifacts: Mapping[str, Path] | None = None,
) -> LoggedVersion:
    """Log the model to the active MLflow run and register it as a new version.

    Must run inside an active run (the training run).
    """
    run = mlflow.active_run()
    if run is None:
        raise RuntimeError("log_version needs an active MLflow run")  # programming error, not user-facing
    artifacts = {"weights": str(weights_dir), "spec": str(spec_path)}
    artifacts.update({k: str(v) for k, v in (extra_artifacts or {}).items()})
    info = mlflow.pyfunc.log_model(
        name=ARTIFACT_PATH,
        python_model=ModelportPyfunc(),
        artifacts=artifacts,
        code_paths=[str(model_dir)],
        model_config={"code_dir": model_dir.name},
        signature=build_signature(spec),
        input_example=example.head(5),
        pip_requirements=list(spec.runtime.pip),
        registered_model_name=registered_model_name,
    )
    version = str(info.registered_model_version)
    _log.info("pyfunc.logged", model=registered_model_name, version=version, run_id=run.info.run_id)
    return LoggedVersion(
        version=version, run_id=run.info.run_id, model_uri=f"models:/{registered_model_name}/{version}"
    )


def load_version(model_uri: str) -> ModelportPyfunc:
    """Load a registered version (`models:/<name>/<version>`) and return its `ModelportPyfunc`.

    Raises:
        NotFoundError: the model or version does not exist.
    """
    start = time.perf_counter()
    try:
        loaded = mlflow.pyfunc.load_model(model_uri)
    except MlflowException as err:
        if getattr(err, "error_code", "") in {"RESOURCE_DOES_NOT_EXIST", "INVALID_PARAMETER_VALUE"}:
            raise NotFoundError(
                what=f"{model_uri} does not exist",
                fix="Check the model name and version with `modelport status`.",
            ) from err
        raise
    python_model: ModelportPyfunc = loaded.unwrap_python_model()
    _log.info("pyfunc.loaded", model_uri=model_uri, seconds=round(time.perf_counter() - start, 3))
    return python_model
