"""Shared helpers for the Airflow DAGs: default arguments, pools, and adapter factories.

DAG files stay thin: each task calls one function here or in a platform module with real adapters, and
tests call the same functions with fakes. Nothing here imports Airflow at module level.
"""

from datetime import datetime, timedelta
from enum import StrEnum
from functools import cache
from typing import TYPE_CHECKING, Any, Final

from modelport.logs import get_logger
from modelport.registry import MlflowRegistry
from modelport_platform.kube import KubeClient, Plural, RayJobPhase, rayjob_status
from modelport_platform.ports import KubeApi
from modelport_platform.s3 import S3DatasetStore, boto3_client
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
        return PokeReturnValue(is_done=True, xcom_value={"startup_s": None, "duration_s": None})
    status = rayjob_status(obj)
    if status.phase is RayJobPhase.FAILED:
        raise AirflowFailException(
            f"RayJob {name} failed ({status.reason}): {status.message}. Logs: kubectl -n ray logs job/{name}"
        )
    if status.phase is not RayJobPhase.SUCCEEDED:
        return PokeReturnValue(is_done=False)
    # shortcut: KubeRay's startTime is when initialising began, so startup (cluster ready) is not known
    # here; duration includes it. Measure startup from the job's own status when timing pushes need it.
    timing = {"startup_s": None, "duration_s": _seconds(status.start_time, status.end_time)}
    return PokeReturnValue(is_done=True, xcom_value=timing)
