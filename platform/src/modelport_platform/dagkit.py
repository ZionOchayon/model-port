"""Shared helpers for the Airflow DAGs: default arguments, pools, and adapter factories.

DAG files stay thin: each task calls one function here or in a platform module with real adapters, and
tests call the same functions with fakes. Nothing here imports Airflow at module level.
"""

from datetime import timedelta
from enum import StrEnum
from functools import cache
from typing import Any, Final

from modelport.registry import MlflowRegistry
from modelport_platform.s3 import S3DatasetStore, boto3_client
from modelport_platform.settings import PlatformSettings

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
