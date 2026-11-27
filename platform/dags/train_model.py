"""train_model: train one submission as a RayJob on its own Ray cluster and register the version (FR-4).

The RayJob runs `modelport_platform.train.entrypoint` with the submitted code as its working_dir; the
entrypoint registers the version, and `record_version` checks that it did.
"""

import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from airflow.sdk import Param, PokeReturnValue, dag, task
from airflow.sdk.exceptions import AirflowFailException

from modelport.errors import ModelportError, SpecError
from modelport.naming import MODEL_NAME_PATTERN, Tag, rayjob_name
from modelport.registry import ArtifactStore, ModelRegistry
from modelport.spec import ModelSpec, load_spec
from modelport_platform import dagkit
from modelport_platform.dagkit import Pool
from modelport_platform.kube import Plural
from modelport_platform.ports import DatasetStore
from modelport_platform.ray.manifests import build_train_rayjob
from modelport_platform.train.data import resolve_dataset_version


def prepare(
    conf: Mapping[str, Any], dag_run_id: str, registry: ArtifactStore, store: DatasetStore, tmp: Path
) -> dict[str, Any]:
    """Read the submission's spec, resolve the dataset version and name the RayJob.

    Raises:
        SpecError: the spec names another model, or the dataset has no such version.
    """
    run_id, spec_file = conf["submission_run_id"], conf["spec_file"]
    spec = load_spec(registry.download(run_id, f"spec/{spec_file}", tmp))
    if spec.name != conf["model"]:
        raise SpecError(
            what=f"{spec_file} of submission {run_id} names '{spec.name}', not '{conf['model']}'",
            fix="Trigger train_model with the model the submission belongs to.",
        )
    version = resolve_dataset_version(store, spec, conf.get("dataset_version"))
    return {
        "rayjob": rayjob_name(spec.name, dag_run_id),
        "working_dir_uri": registry.artifact_s3_uri(run_id, "code/code.zip"),
        "dataset_version": version.version,
        "spec_json": spec.model_dump_json(by_alias=True),
    }


def record_version(registry: ModelRegistry, model: str, rayjob: str) -> str:
    """The version the RayJob registered; fail when there is none."""
    info = registry.find_version_by_tag(model, Tag.TRAIN_RAYJOB, rayjob)
    if info is None:
        raise AirflowFailException(
            f"{model}: training finished but registered no version (RayJob {rayjob}). "
            f"Logs: kubectl -n ray logs job/{rayjob}"
        )
    return info.version


@dag(
    dag_id="train_model",
    schedule=None,
    max_active_runs=2,
    catchup=False,
    default_args=dagkit.DEFAULT_ARGS,
    tags=[*dagkit.TAGS, "train"],
    doc_md=__doc__,
    params={
        "model": Param(type="string", pattern=MODEL_NAME_PATTERN),
        "submission_run_id": Param(type="string", minLength=1),
        "dataset_version": Param(None, type=["null", "integer"], minimum=1),
        "spec_file": Param("model.yaml", type="string", pattern=r"^[\w.-]+\.ya?ml$"),
    },
)
def train_model() -> None:
    """Prepare -> create_rayjob -> wait_rayjob -> record_version."""

    @task(task_id="prepare")
    def prepare_task(params: Any = None, run_id: Any = None) -> dict[str, Any]:  # noqa: ANN401
        with tempfile.TemporaryDirectory() as tmp:
            try:
                return prepare(params, run_id, dagkit.registry(), dagkit.store(), Path(tmp))
            except ModelportError as err:  # a bad spec or missing data: retrying cannot help
                raise AirflowFailException(f"{err.what} How to fix: {err.fix}") from err

    @task(pool=Pool.RAY_TRAINING)
    def create_rayjob(prepared: dict[str, Any], params: Any = None) -> str:  # noqa: ANN401
        body = build_train_rayjob(
            name=prepared["rayjob"],
            spec=ModelSpec.model_validate_json(prepared["spec_json"]),
            submission_run_id=params["submission_run_id"],
            spec_file=params["spec_file"],
            dataset_version=prepared["dataset_version"],
            working_dir_uri=prepared["working_dir_uri"],
            settings=dagkit.settings(),
        )
        dagkit.kube().ensure(Plural.RAYJOBS, body)  # a retry adopts the existing RayJob
        return str(prepared["rayjob"])

    @task.sensor(mode="reschedule", poke_interval=15, timeout=7200, pool=Pool.RAY_TRAINING)
    def wait_rayjob(rayjob: str) -> PokeReturnValue:
        return dagkit.rayjob_poke(dagkit.kube(), rayjob)

    @task(task_id="record_version")
    def record_version_task(rayjob: str, params: Any = None) -> str:  # noqa: ANN401
        return record_version(dagkit.registry(), params["model"], rayjob)

    rayjob = create_rayjob(prepare_task())
    wait_rayjob(rayjob) >> record_version_task(rayjob)


DAG = train_model()
