"""Shared helpers for the Airflow DAGs: default arguments, pools, and adapter factories.

DAG files stay thin: each task calls one function here or in a platform module with real adapters, and
tests call the same functions with fakes. Nothing here imports Airflow at module level.
"""

import tempfile
from collections.abc import Collection
from datetime import datetime, timedelta
from enum import StrEnum
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, Protocol

from modelport.logs import get_logger
from modelport.naming import Tag
from modelport.registry import ArtifactStore, MlflowRegistry, ModelRegistry, VersionInfo
from modelport.spec import ModelSpec, load_spec
from modelport_platform.kube import KubeClient, Plural, RayJobPhase, rayjob_status, rayservice_apps_ready
from modelport_platform.ports import KubeApi
from modelport_platform.ray.jobs import SharedClusterJobs
from modelport_platform.ray.manifests import build_rayservice
from modelport_platform.s3 import S3DatasetStore, boto3_client
from modelport_platform.serve.config import ServedModel, render_serve_config, served_model_from_spec
from modelport_platform.settings import PlatformSettings

if TYPE_CHECKING:
    from airflow.sdk import PokeReturnValue

_log = get_logger(__name__)

DEFAULT_ARGS: Final[dict[str, Any]] = {
    "owner": "modelport",
    "retries": 3,
    "retry_delay": timedelta(seconds=30),
    "retry_exponential_backoff": True,
    "max_retry_delay": timedelta(minutes=5),
}
TAGS: Final = ["modelport"]


class Pool(StrEnum):
    """Airflow pools (created by `make up`)."""

    RAY_TRAINING = "ray_training"  # slots: smoke 1, full 2 - concurrent training RayJobs
    SHARED_CLUSTER = "shared_cluster"  # slots: smoke 1, full 2 - eval and batch jobs
    SERVE_CONFIG = "serve_config"  # slots: 1 - every task that changes the RayService or serve intent tags


@cache
def settings() -> PlatformSettings:
    """Platform settings from the task's environment (cached per process)."""
    return PlatformSettings.from_env()


def registry() -> MlflowRegistry:
    """The MLflow registry adapter."""
    return MlflowRegistry(settings().mlflow_uri)


def store() -> S3DatasetStore:
    """The S3 dataset store adapter."""
    return S3DatasetStore(boto3_client(settings()))


def jobs() -> SharedClusterJobs:
    """Ray Jobs on the shared RayCluster."""
    return SharedClusterJobs(settings().shared_ray_address)


def spec_for_version(store: ArtifactStore, info: VersionInfo) -> ModelSpec:
    """The spec a version was trained with (`spec/model.yaml` in its training run)."""
    with tempfile.TemporaryDirectory() as tmp:
        return load_spec(store.download(info.tags[Tag.TRAIN_RUN_ID], "spec/model.yaml", Path(tmp)))


def kube() -> KubeClient:
    """The Kubernetes adapter for the Ray namespace."""
    return KubeClient(settings().ray_namespace)


def _seconds(start: datetime | None, end: datetime | None) -> float | None:
    return (end - start).total_seconds() if start and end else None


def rayjob_poke(k: KubeApi, name: str) -> "PokeReturnValue":
    """Sensor poke for a RayJob: done on success (XCom: startup and run seconds), fail on failure.

    A RayJob already deleted by its TTL counts as done: the next task checks that a version exists.
    """
    from airflow.sdk import PokeReturnValue
    from airflow.sdk.exceptions import AirflowFailException

    obj = k.get(Plural.RAYJOBS, name)
    if obj is None:
        _log.warning("rayjob.missing", rayjob=name)
        return PokeReturnValue(
            is_done=True, xcom_value={"job_status": None, "startup_s": None, "duration_s": None}
        )
    status = rayjob_status(obj)
    if status.phase is RayJobPhase.FAILED:
        raise AirflowFailException(
            f"RayJob {name} failed ({status.reason}): {status.message}. Logs: kubectl -n ray logs job/{name}"
        )
    if status.phase is not RayJobPhase.SUCCEEDED:
        return PokeReturnValue(is_done=False)
    # shortcut: KubeRay's startTime is when initialising began, so startup (cluster ready) is not known
    # here; duration includes it. Measure startup from the job's own status when timing pushes need it.
    timing = {
        "job_status": status.job_status,
        "startup_s": None,
        "duration_s": _seconds(status.start_time, status.end_time),
    }
    return PokeReturnValue(is_done=True, xcom_value=timing)


class ServeRegistry(ModelRegistry, ArtifactStore, Protocol):
    """What rendering the serve state reads: aliases, model tags, versions and their training specs."""


def desired_serve_state(reg: ServeRegistry) -> list[ServedModel]:
    """Every model the RayService should serve, from the registry alone (aliases and serve tags).

    A model is served when it has a `champion`, or a serve candidate (`modelport.serve.candidate_version`,
    set by deploy_candidate or serve_sync; the `candidate` alias alone does not serve a version). The spec
    of the champion (else of the candidate) gives resources and thresholds.
    """
    served = []
    for name in reg.list_models():
        snapshot = reg.snapshot(name)
        tags = reg.model_tags(name)
        candidate_version = tags.get(Tag.SERVE_CANDIDATE_VERSION) or None
        if snapshot.champion is None and candidate_version is None:
            continue
        champion = reg.get_version(name, snapshot.champion) if snapshot.champion else None
        candidate = reg.get_version(name, candidate_version) if candidate_version else None
        owner = champion or candidate
        assert owner is not None
        served.append(
            served_model_from_spec(
                spec_for_version(reg, owner),
                champion,
                candidate,
                candidate_weight=float(tags.get(Tag.SERVE_CANDIDATE_WEIGHT) or 0.0),
                fault_error_rate=float(tags.get(Tag.SERVE_FAULT_ERROR_RATE) or 0.0),
            )
        )
    return served


def apply_serve_state(reg: ServeRegistry, k: KubeApi, s: PlatformSettings) -> list[str]:
    """Render the whole serve config and write it to the RayService; returns the app names to wait for.

    Runs only in pool `serve_config`, so two DAGs never write competing configs.
    """
    config = render_serve_config(desired_serve_state(reg), env_vars={"MLFLOW_TRACKING_URI": s.mlflow_uri})
    body = build_rayservice(serve_config=config, settings=s)
    _, created = k.ensure(Plural.RAYSERVICES, body)
    if not created:
        k.merge_patch(
            Plural.RAYSERVICES, s.rayservice_name, {"spec": {"serveConfigV2": body["spec"]["serveConfigV2"]}}
        )
    apps = [str(app["name"]) for app in config["applications"]]
    _log.info("serve.applied", created=created, apps=apps)
    return apps


def rayservice_ready_poke(k: KubeApi, apps: Collection[str], *, name: str | None = None) -> "PokeReturnValue":
    """Sensor poke: done when the RayService runs every app of its current generation.

    Raises:
        AirflowFailException: an expected app is DEPLOY_FAILED (fail fast instead of waiting it out).
    """
    from airflow.sdk import PokeReturnValue
    from airflow.sdk.exceptions import AirflowFailException

    obj = k.get(Plural.RAYSERVICES, name or settings().rayservice_name)
    if obj is None:
        return PokeReturnValue(is_done=False)
    ready, failure = rayservice_apps_ready(obj, apps)
    if failure is not None:
        raise AirflowFailException(failure)
    return PokeReturnValue(is_done=ready)
