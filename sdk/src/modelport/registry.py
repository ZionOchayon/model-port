"""The only code that talks to MLflow: versions, aliases, tags, submissions and artifacts (Repository).

`ModelRegistry` and `ArtifactStore` are the ports; `MlflowRegistry` implements both over `MlflowClient`.
Logical MLflow errors become modelport errors; network failures are retried, then reported as
`PlatformUnavailableError`.
"""

import json
import os
import re
import tempfile
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from typing import Any, Final, Protocol

from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

from modelport.errors import NotFoundError, PlatformUnavailableError
from modelport.lifecycle import AliasOpKind, AliasSnapshot, ModelState, Plan
from modelport.logs import get_logger
from modelport.naming import EXPERIMENT_SUBMISSIONS, Alias, Tag, submission_key
from modelport.packaging import CodePackage

RETRY_DELAYS_S: Final = (1, 2, 4)
_LOGICAL_ERRORS: Final = frozenset(
    {
        "RESOURCE_DOES_NOT_EXIST",
        "RESOURCE_ALREADY_EXISTS",
        "INVALID_PARAMETER_VALUE",
        "INVALID_STATE",
        "BAD_REQUEST",
    }
)
_SAFE_FILTER_VALUE = re.compile(r"[A-Za-z0-9_.:/-]*")
_log = get_logger(__name__)


@dataclass(frozen=True)
class VersionInfo:
    """One registered model version as modelport sees it."""

    name: str
    version: str
    run_id: str
    source: str
    tags: Mapping[str, str]
    aliases: tuple[str, ...]
    created_at: datetime

    @property
    def state(self) -> ModelState:
        """From tag `modelport.state`; a version without it is `registered`."""
        return ModelState(self.tags.get(Tag.STATE, ModelState.REGISTERED))


class ModelRegistry(Protocol):
    """Registered models, versions, aliases and tags."""

    def ensure_registered_model(self, name: str, tags: Mapping[str, str]) -> None:
        """Create the registered model if needed and set its tags."""

    def get_version(self, name: str, version: str) -> VersionInfo:
        """One version; NotFoundError if it does not exist."""

    def resolve(self, name: str, alias: Alias) -> VersionInfo | None:
        """The version an alias points to, or None."""

    def snapshot(self, name: str) -> AliasSnapshot:
        """The three aliases and the states of the versions they name."""

    def find_version_by_tag(self, name: str, key: Tag, value: str) -> VersionInfo | None:
        """The newest version whose tag equals value."""

    def set_version_tags(self, name: str, version: str, tags: Mapping[Tag, str]) -> None:
        """Set tags on one version."""

    def set_model_tags(self, name: str, tags: Mapping[Tag, str]) -> None:
        """Set tags on the registered model."""

    def model_tags(self, name: str) -> Mapping[str, str]:
        """Tags of the registered model; NotFoundError if it does not exist."""

    def apply_plan(self, name: str, plan: Plan) -> None:
        """Execute a lifecycle plan (alias operations in order, then states)."""

    def list_models(self, *, tag_equals: Mapping[Tag, str] | None = None) -> list[str]:
        """Registered model names whose tags match, sorted."""

    def model_uri(self, name: str, version: str) -> str:
        """`models:/<name>/<version>`."""

    def latest_version(self, name: str) -> VersionInfo | None:
        """The highest-numbered version, or None for a model without versions."""


class ArtifactStore(Protocol):
    """Submission runs and run artifacts."""

    def find_submission(self, model: str, submission_key: str) -> str | None:
        """The run ID of an earlier identical submission, or None."""

    def create_submission(self, model: str, package: CodePackage, spec_path: Path, spec_sha256: str) -> str:
        """Upload the code and spec as a new submission run; return its run ID."""

    def download(self, run_id: str, artifact_path: str, dst: Path) -> Path:
        """Download one artifact into dst."""

    def log_json(self, run_id: str, artifact_path: str, payload: Mapping[str, Any]) -> None:
        """Write a JSON artifact."""

    def read_json(self, run_id: str, artifact_path: str) -> dict[str, Any]:
        """Read a JSON artifact."""

    def artifact_s3_uri(self, run_id: str, artifact_path: str) -> str:
        """The S3 location of an artifact."""

    def run_url(self, run_id: str) -> str:
        """Link to the run in the MLflow UI."""

    def run_tags(self, run_id: str) -> Mapping[str, str]:
        """All tags of one run."""


def _code(err: MlflowException) -> str:
    return str(getattr(err, "error_code", ""))


def _quoted(value: str) -> str:
    if not _SAFE_FILTER_VALUE.fullmatch(value):
        raise ValueError(f"unsafe value in an MLflow filter: {value!r}")
    return f"'{value}'"


class MlflowRegistry:
    """`ModelRegistry` and `ArtifactStore` over `MlflowClient` (Adapter, Repository)."""

    def __init__(
        self,
        tracking_uri: str,
        *,
        client: "MlflowClient | None" = None,
        ui_base_url: str | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.tracking_uri = tracking_uri
        self.client = client or MlflowClient(tracking_uri=tracking_uri, registry_uri=tracking_uri)
        self.ui_base_url = (ui_base_url or "http://localhost:5000").rstrip("/")
        self._sleep = sleep

    def _call[T](self, fn: Callable[[], T]) -> T:
        """Run one MLflow call. Logical errors raise at once; transport failures are retried 3 times."""
        last: Exception | None = None
        for delay in (*RETRY_DELAYS_S, None):
            try:
                return fn()
            except MlflowException as err:
                if _code(err) in _LOGICAL_ERRORS:
                    raise
                last = err
            except (ConnectionError, TimeoutError, OSError) as err:
                last = err
            if delay is None:
                break
            _log.debug("registry.retry", delay_s=delay, uri=self.tracking_uri, error=type(last).__name__)
            self._sleep(delay)
        raise PlatformUnavailableError(
            what=f"Cannot reach MLflow at {self.tracking_uri}",
            fix="Run `make up` and `make ui`, or set MLFLOW_TRACKING_URI.",
        ) from last

    # ModelRegistry

    def ensure_registered_model(self, name: str, tags: Mapping[str, str]) -> None:
        """Create the registered model if needed (a concurrent create is fine), then set its tags."""
        try:
            self._call(lambda: self.client.get_registered_model(name))
        except MlflowException as err:
            if _code(err) != "RESOURCE_DOES_NOT_EXIST":
                raise
            try:
                self._call(lambda: self.client.create_registered_model(name, tags=dict(tags)))
            except MlflowException as race:
                if _code(race) != "RESOURCE_ALREADY_EXISTS":
                    raise
        for key, value in tags.items():
            self._call(partial(self.client.set_registered_model_tag, name, key, value))

    def _info(self, mv: Any) -> VersionInfo:  # noqa: ANN401 - mlflow ModelVersion
        return VersionInfo(
            name=mv.name,
            version=str(mv.version),
            run_id=mv.run_id or "",
            source=mv.source or "",
            tags=dict(mv.tags or {}),
            aliases=tuple(mv.aliases or ()),
            created_at=datetime.fromtimestamp(mv.creation_timestamp / 1000, tz=UTC),
        )

    def get_version(self, name: str, version: str) -> VersionInfo:
        """One version.

        Raises:
            NotFoundError: the model or version does not exist.
        """
        try:
            return self._info(self._call(lambda: self.client.get_model_version(name, version)))
        except MlflowException as err:
            if _code(err) in {"RESOURCE_DOES_NOT_EXIST", "INVALID_PARAMETER_VALUE"}:
                raise NotFoundError(
                    what=f"{name} has no version {version}",
                    fix=f"List the versions with `modelport status {name}`.",
                ) from err
            raise

    def resolve(self, name: str, alias: Alias) -> VersionInfo | None:
        """The version an alias points to, or None when the alias is not set."""
        try:
            return self._info(self._call(lambda: self.client.get_model_version_by_alias(name, alias.value)))
        except MlflowException as err:
            if _code(err) in {"RESOURCE_DOES_NOT_EXIST", "INVALID_PARAMETER_VALUE"}:
                return None
            raise

    def snapshot(self, name: str) -> AliasSnapshot:
        """The three aliases and the states of the versions they name.

        Raises:
            NotFoundError: no registered model has this name.
        """
        try:
            model = self._call(lambda: self.client.get_registered_model(name))
        except MlflowException as err:
            if _code(err) == "RESOURCE_DOES_NOT_EXIST":
                raise NotFoundError(
                    what=f"No model named '{name}' is registered",
                    fix="Check the name with `modelport status`; names come from model.yaml.",
                ) from err
            raise
        aliases = {str(alias): str(version) for alias, version in (model.aliases or {}).items()}
        states = {v: self.get_version(name, v).state for v in set(aliases.values())}
        return AliasSnapshot(
            model=name,
            champion=aliases.get(Alias.CHAMPION),
            previous=aliases.get(Alias.PREVIOUS),
            candidate=aliases.get(Alias.CANDIDATE),
            states=states,
        )

    def find_version_by_tag(self, name: str, key: Tag, value: str) -> VersionInfo | None:
        """The newest version whose tag `key` equals `value`."""
        query = f"name = {_quoted(name)} and tags.`{key}` = {_quoted(value)}"
        found = self._call(lambda: self.client.search_model_versions(query, max_results=50))
        if not found:
            return None
        return self._info(max(found, key=lambda mv: int(mv.version)))

    def latest_version(self, name: str) -> VersionInfo | None:
        """The highest-numbered version, or None."""
        query = f"name = {_quoted(name)}"
        order = ["version_number DESC"]
        found = self._call(lambda: self.client.search_model_versions(query, order_by=order, max_results=1))
        return self._info(found[0]) if found else None

    def set_version_tags(self, name: str, version: str, tags: Mapping[Tag, str]) -> None:
        """Set tags on one version."""
        for key, value in tags.items():
            self._call(partial(self.client.set_model_version_tag, name, version, key, value))

    def set_model_tags(self, name: str, tags: Mapping[Tag, str]) -> None:
        """Set tags on the registered model."""
        for key, value in tags.items():
            self._call(partial(self.client.set_registered_model_tag, name, key, value))

    def model_tags(self, name: str) -> Mapping[str, str]:
        """Tags of the registered model.

        Raises:
            NotFoundError: no registered model has this name.
        """
        try:
            model = self._call(lambda: self.client.get_registered_model(name))
        except MlflowException as err:
            if _code(err) == "RESOURCE_DOES_NOT_EXIST":
                raise NotFoundError(
                    what=f"No model named '{name}' is registered",
                    fix="Check the name with `modelport status`.",
                ) from err
            raise
        return dict(model.tags or {})

    def apply_plan(self, name: str, plan: Plan) -> None:
        """Run the alias operations in order, then write the new states. Re-applying converges."""
        for op in plan.alias_ops:
            if op.kind is AliasOpKind.SET and op.version is not None:
                self._call(partial(self.client.set_registered_model_alias, name, op.alias.value, op.version))
                _log.info("registry.alias_set", model=name, alias=op.alias.value, version=op.version)
            else:
                try:
                    self._call(partial(self.client.delete_registered_model_alias, name, op.alias.value))
                except MlflowException as err:
                    if _code(err) not in {"RESOURCE_DOES_NOT_EXIST", "INVALID_PARAMETER_VALUE"}:
                        raise
                _log.info("registry.alias_deleted", model=name, alias=op.alias.value)
        stamp = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
        for version, state in plan.states.items():
            self.set_version_tags(name, version, {Tag.STATE: state.value, Tag.STATE_CHANGED_AT: stamp})
            _log.info("registry.state_set", model=name, version=version, state=state.value)

    def list_models(self, *, tag_equals: Mapping[Tag, str] | None = None) -> list[str]:
        """Names of registered models whose tags equal `tag_equals`, sorted."""
        query = " and ".join(f"tags.`{k}` = {_quoted(v)}" for k, v in (tag_equals or {}).items()) or None
        names: list[str] = []
        token: str | None = None
        while True:
            page = self._call(
                partial(
                    self.client.search_registered_models,
                    filter_string=query,
                    max_results=100,
                    page_token=token,
                )
            )
            names += [m.name for m in page]
            token = page.token
            if not token:
                return sorted(names)

    def model_uri(self, name: str, version: str) -> str:
        """`models:/<name>/<version>`."""
        return f"models:/{name}/{version}"

    def run_tags(self, run_id: str) -> dict[str, str]:
        """All tags of one run."""
        return dict(self._call(lambda: self.client.get_run(run_id)).data.tags)

    def ensure_run(
        self, experiment: str, run_name: str, key_tag: Tag, tags: Mapping[str, str], params: Mapping[str, Any]
    ) -> str:
        """Find the run tagged `key_tag` = `tags[key_tag]` or create it; then set tags and params."""
        exp = self._call(lambda: self.client.get_experiment_by_name(experiment))
        experiment_id = (
            str(exp.experiment_id)
            if exp
            else str(self._call(lambda: self.client.create_experiment(experiment)))
        )
        query = f"tags.`{key_tag}` = {_quoted(tags[key_tag])}"
        found = self._call(lambda: self.client.search_runs([experiment_id], query, max_results=1))
        if found:
            run_id = str(found[0].info.run_id)
        else:
            run_id = str(
                self._call(lambda: self.client.create_run(experiment_id, run_name=run_name)).info.run_id
            )
        for key, value in tags.items():
            self._call(partial(self.client.set_tag, run_id, key, value))
        for key, value in params.items():
            try:
                self._call(partial(self.client.log_param, run_id, key, value))
            except MlflowException as err:  # params are immutable: a resumed run keeps its first values
                if _code(err) != "INVALID_PARAMETER_VALUE":
                    raise
        return run_id

    def log_metrics(self, run_id: str, metrics: Mapping[str, float], step: int) -> None:
        """Log metrics to a run at one step (used by `modelport.report` on rank 0)."""
        for key, value in metrics.items():
            self._call(partial(self.client.log_metric, run_id, key, value, step=step))

    # ArtifactStore

    def _submissions_experiment(self, *, create: bool) -> str | None:
        exp = self._call(lambda: self.client.get_experiment_by_name(EXPERIMENT_SUBMISSIONS))
        if exp is not None:
            return str(exp.experiment_id)
        if not create:
            return None
        try:
            return str(self._call(lambda: self.client.create_experiment(EXPERIMENT_SUBMISSIONS)))
        except MlflowException as err:
            if _code(err) != "RESOURCE_ALREADY_EXISTS":
                raise
            return self._submissions_experiment(create=False)

    def find_submission(self, model: str, submission_key: str) -> str | None:
        """The newest submission run of `model` with this key, or None."""
        experiment = self._submissions_experiment(create=False)
        if experiment is None:
            return None
        model_match = f"tags.`{Tag.MODEL}` = {_quoted(model)}"
        query = f"{model_match} and tags.`{Tag.SUBMISSION_KEY}` = {_quoted(submission_key)}"
        runs = self._call(
            lambda: self.client.search_runs(
                [experiment], query, max_results=1, order_by=["attributes.start_time DESC"]
            )
        )
        return runs[0].info.run_id if runs else None

    def create_submission(self, model: str, package: CodePackage, spec_path: Path, spec_sha256: str) -> str:
        """A finished run in `modelport-submissions` holding `code/code.zip` and `spec/<spec file>`."""
        experiment = self._submissions_experiment(create=True)
        assert experiment is not None
        tags = {
            Tag.MODEL: model,
            Tag.CODE_SHA256: package.sha256,
            Tag.SPEC_SHA256: spec_sha256,
            Tag.SUBMISSION_KEY: submission_key(package.sha256, spec_path.name, spec_sha256),
        }
        run = self._call(
            lambda: self.client.create_run(
                experiment,
                run_name=f"{model}-{package.sha256[:12]}",
                tags={k.value: v for k, v in tags.items()},
            )
        )
        run_id: str = run.info.run_id
        self._call(lambda: self.client.log_artifact(run_id, str(package.path), "code"))
        self._call(lambda: self.client.log_artifact(run_id, str(spec_path), "spec"))
        self._call(lambda: self.client.set_terminated(run_id, "FINISHED"))
        return run_id

    def download(self, run_id: str, artifact_path: str, dst: Path) -> Path:
        """Download one artifact into `dst` and return its local path."""
        dst.mkdir(parents=True, exist_ok=True)
        try:
            return Path(self._call(lambda: self.client.download_artifacts(run_id, artifact_path, str(dst))))
        except (MlflowException, OSError) as err:
            if isinstance(err, MlflowException) and _code(err) not in _LOGICAL_ERRORS:
                raise
            raise NotFoundError(
                what=f"Run {run_id} has no artifact {artifact_path}",
                fix="The run may be incomplete; check it in the MLflow UI.",
            ) from err

    def log_json(self, run_id: str, artifact_path: str, payload: Mapping[str, Any]) -> None:
        """Write `payload` as a JSON artifact (overwrites an earlier one at the same path)."""
        self._call(lambda: self.client.log_dict(run_id, dict(payload), artifact_path))

    def read_json(self, run_id: str, artifact_path: str) -> dict[str, Any]:
        """Read a JSON artifact."""
        with tempfile.TemporaryDirectory(prefix="modelport-") as tmp:
            data: dict[str, Any] = json.loads(
                self.download(run_id, artifact_path, Path(tmp)).read_text("utf-8")
            )
        return data

    def artifact_s3_uri(self, run_id: str, artifact_path: str) -> str:
        """Where the server stores an artifact (`mlflow-artifacts:/…` maps to `s3://mlflow/…`)."""
        root = str(self._call(lambda: self.client.get_run(run_id)).info.artifact_uri)
        root = re.sub(r"^mlflow-artifacts:/*", "s3://mlflow/", root)
        return f"{root.rstrip('/')}/{artifact_path}"

    def run_url(self, run_id: str) -> str:
        """Link to the run in the MLflow UI."""
        run = self._call(lambda: self.client.get_run(run_id))
        return f"{self.ui_base_url}/#/experiments/{run.info.experiment_id}/runs/{run_id}"


def registry_from_env() -> MlflowRegistry:
    """`MlflowRegistry` from MLFLOW_TRACKING_URI and MODELPORT_MLFLOW_UI_URL.

    MLflow's own HTTP retries are turned off unless set, so `MlflowRegistry`'s policy (3 retries) applies.
    """
    os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "0")
    return MlflowRegistry(
        os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"),
        ui_base_url=os.environ.get("MODELPORT_MLFLOW_UI_URL"),
    )
