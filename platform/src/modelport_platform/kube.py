"""Idempotent create, read, patch and delete of KubeRay custom objects (Adapter for the KubeApi port).

Also the pure readers of KubeRay status: `rayjob_status` and `rayservice_apps_ready`.
"""

from collections.abc import Collection, Mapping
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Any, Final

from kubernetes import client as k8s
from kubernetes import config as k8s_config

from modelport.errors import PlatformUnavailableError
from modelport.logs import get_logger

RAY_GROUP: Final = "ray.io"
RAY_VERSION: Final = "v1"
_log = get_logger(__name__)


class Plural(StrEnum):
    """KubeRay resources modelport manages."""

    RAYJOBS = "rayjobs"
    RAYSERVICES = "rayservices"
    RAYCLUSTERS = "rayclusters"


class KubeClient:
    """`KubeApi` over the Kubernetes `CustomObjectsApi`, in one namespace."""

    def __init__(self, namespace: str, api: Any = None) -> None:  # noqa: ANN401 - kubernetes ships no stubs
        self.namespace = namespace
        if api is None:
            try:
                k8s_config.load_incluster_config()
            except k8s_config.ConfigException:
                k8s_config.load_kube_config()
            api = k8s.CustomObjectsApi()
        self._api = api

    def _call(self, fn: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        try:
            return fn(group=RAY_GROUP, version=RAY_VERSION, namespace=self.namespace, **kwargs)
        except k8s.ApiException as err:
            if err.status in (401, 403):
                raise PlatformUnavailableError(
                    what="Airflow may not manage Ray resources",
                    fix="Check deploy/manifests/base/airflow-rbac.yaml was applied (make up).",
                ) from err
            raise

    def ensure(self, plural: str, body: Mapping[str, Any]) -> tuple[dict[str, Any], bool]:
        """Create the object; if it exists (409), return the existing one unchanged."""
        name = body["metadata"]["name"]
        try:
            created: dict[str, Any] = self._call(
                self._api.create_namespaced_custom_object, plural=plural, body=dict(body)
            )
        except k8s.ApiException as err:
            if err.status != 409:
                raise
            existing = self.get(plural, name)
            if existing is None:  # deleted between the create and the get: create again
                return self.ensure(plural, body)
            _log.info("kube.adopted", plural=plural, name=name)
            return existing, False
        _log.info("kube.created", plural=plural, name=name)
        return created, True

    def get(self, plural: str, name: str) -> dict[str, Any] | None:
        """The object, or None when it does not exist."""
        try:
            obj: dict[str, Any] = self._call(self._api.get_namespaced_custom_object, plural=plural, name=name)
        except k8s.ApiException as err:
            if err.status == 404:
                return None
            raise
        return obj

    def merge_patch(self, plural: str, name: str, patch: Mapping[str, Any]) -> dict[str, Any]:
        """Apply a JSON merge patch (the client sends merge-patch for a dict body)."""
        obj: dict[str, Any] = self._call(
            self._api.patch_namespaced_custom_object, plural=plural, name=name, body=dict(patch)
        )
        _log.info("kube.patched", plural=plural, name=name)
        return obj

    def delete(self, plural: str, name: str) -> None:
        """Delete the object; a missing object is fine."""
        try:
            self._call(self._api.delete_namespaced_custom_object, plural=plural, name=name)
        except k8s.ApiException as err:
            if err.status != 404:
                raise
        _log.info("kube.deleted", plural=plural, name=name)


class RayJobPhase(StrEnum):
    """modelport's view of a RayJob."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


@dataclass(frozen=True)
class RayJobStatus:
    """The fields of a RayJob's status that the DAGs use."""

    phase: RayJobPhase
    job_status: str | None
    deployment_status: str | None
    reason: str | None
    message: str | None
    start_time: datetime | None
    end_time: datetime | None
    cluster_name: str | None


def _time(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value.replace("Z", "+00:00")) if value else None


def rayjob_status(obj: Mapping[str, Any]) -> RayJobStatus:
    """Map KubeRay's `jobDeploymentStatus` x `jobStatus` to one phase (read both: they lag)."""
    status = obj.get("status") or {}
    deployment, job = status.get("jobDeploymentStatus") or "", status.get("jobStatus") or ""
    if deployment == "Complete" and job == "SUCCEEDED":
        phase = RayJobPhase.SUCCEEDED
    elif deployment in {"Failed", "ValidationFailed"} or job in {"FAILED", "STOPPED"}:
        phase = RayJobPhase.FAILED
    elif deployment == "Running":
        phase = RayJobPhase.RUNNING
    else:
        phase = RayJobPhase.PENDING
    return RayJobStatus(
        phase=phase,
        job_status=job or None,
        deployment_status=deployment or None,
        reason=status.get("reason"),
        message=status.get("message"),
        start_time=_time(status.get("startTime")),
        end_time=_time(status.get("endTime")),
        cluster_name=status.get("rayClusterName"),
    )


def rayservice_apps_ready(obj: Mapping[str, Any], expected_apps: Collection[str]) -> tuple[bool, str | None]:
    """(ready, failure): ready once the current generation runs every app; a failure names DEPLOY_FAILED.

    Readiness is `serviceStatus: Running` or (KubeRay 1.7, where serviceStatus may be absent) a `Ready`
    condition. A DEPLOY_FAILED app is reported from the active status or, during a RayService's first
    deployment, from the pending one.
    """
    status = obj.get("status") or {}
    apps = (status.get("activeServiceStatus") or {}).get("applicationStatuses") or {}
    pending = (status.get("pendingServiceStatus") or {}).get("applicationStatuses") or {}
    for name in expected_apps:
        for source in (apps, pending):
            app = source.get(name) or {}
            if app.get("status") == "DEPLOY_FAILED":
                return False, f"{name} DEPLOY_FAILED: {app.get('message', '')}"
    conditions = {c.get("type"): c.get("status") for c in status.get("conditions") or []}
    serving = status.get("serviceStatus") == "Running" or conditions.get("Ready") == "True"
    ready = (
        status.get("observedGeneration") == (obj.get("metadata") or {}).get("generation")
        and serving
        and all((apps.get(name) or {}).get("status") == "RUNNING" for name in expected_apps)
    )
    return ready, None
