"""Build complete KubeRay objects as plain dicts (Builder; pure, snapshot-tested).

The head always has `num-cpus: 0`, so every Train worker gets its own worker pod (DDP on N workers).
"""

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Final

import yaml

from modelport.spec import ModelSpec
from modelport_platform.settings import PlatformSettings, Profile

RAY_VERSION: Final = "2.60.1"
TRAIN_SLOT: Final = "modelport_train_slot"  # custom Ray resource: one Train worker per worker pod
RAY_POD_ENV: Final[list[dict[str, str]]] = [{"name": "RAY_TRAIN_V2_ENABLED", "value": "1"}]
_PART_OF: Final = {"app.kubernetes.io/part-of": "modelport"}


@dataclass(frozen=True)
class PodSizing:
    """Requests and limits of one container."""

    cpu_request: str
    cpu_limit: str
    memory_request: str
    memory_limit: str

    def resources(self) -> dict[str, dict[str, str]]:
        """The Kubernetes `resources` block."""
        return {
            "requests": {"cpu": self.cpu_request, "memory": self.memory_request},
            "limits": {"cpu": self.cpu_limit, "memory": self.memory_limit},
        }


SUBMITTER: Final = PodSizing("100m", "1", "200Mi", "1Gi")


def _cpu(cores: float) -> str:
    """1.0 -> "1", 0.5 -> "500m"."""
    return str(int(cores)) if float(cores).is_integer() else f"{round(cores * 1000)}m"


def head_sizing(profile: Profile) -> PodSizing:
    """Ray head of any cluster (section 9.7)."""
    return (
        PodSizing("200m", "1", "1536Mi", "3Gi")
        if profile is Profile.SMOKE
        else PodSizing("250m", "1", "2Gi", "4Gi")
    )


def train_worker_sizing(spec: ModelSpec, profile: Profile) -> PodSizing:
    """One Train worker: `cpus_per_worker` and `memory_per_worker`; smoke requests 250m CPU."""
    train = spec.resources.train
    request = "250m" if profile is Profile.SMOKE else _cpu(train.cpus_per_worker)
    return PodSizing(request, _cpu(train.cpus_per_worker), train.memory_per_worker, train.memory_per_worker)


def serve_worker_sizing(profile: Profile) -> PodSizing:
    """A Serve worker pod."""
    return (
        PodSizing("500m", "2", "2560Mi", "3584Mi")
        if profile is Profile.SMOKE
        else PodSizing("1", "2", "3Gi", "4Gi")
    )


def ray_pod_env_from() -> list[dict[str, Any]]:
    """Every Ray pod reads the platform ConfigMap and the S3 Secret."""
    return [{"configMapRef": {"name": "modelport-platform-env"}}, {"secretRef": {"name": "s3-credentials"}}]


def _container(name: str, image: str, sizing: PodSizing, ports: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "name": name,
        "image": image,
        "imagePullPolicy": "IfNotPresent",
        "envFrom": ray_pod_env_from(),
        "env": RAY_POD_ENV,
        "ports": ports,
        "resources": sizing.resources(),
    }


HEAD_PORTS: Final = [
    {"containerPort": 6379, "name": "gcs"},
    {"containerPort": 8265, "name": "dashboard"},
    {"containerPort": 10001, "name": "client"},
    {"containerPort": 8080, "name": "metrics"},
]
WORKER_PORTS: Final = [{"containerPort": 8080, "name": "metrics"}]
# KubeRay creates the Serve Service only when the head container has a port named `serve`
SERVE_HEAD_PORTS: Final = [*HEAD_PORTS, {"containerPort": 8000, "name": "serve"}]


def _cluster_spec(
    *,
    labels: dict[str, str],
    image: str,
    head: PodSizing,
    group_name: str,
    workers: int,
    worker_cpus: int,
    worker: PodSizing,
    ray_resources: dict[str, str] | None = None,
    head_ports: list[dict[str, Any]] = HEAD_PORTS,
) -> dict[str, Any]:
    group: dict[str, Any] = {
        "groupName": group_name,
        "replicas": workers,
        "minReplicas": workers,
        "maxReplicas": workers,
        "rayStartParams": {"metrics-export-port": "8080"},
    }
    if ray_resources is None:
        group["rayStartParams"]["num-cpus"] = str(worker_cpus)
    else:
        group["resources"] = ray_resources  # KubeRay group field; overrides rayStartParams
    group["template"] = {
        "metadata": {"labels": labels},
        "spec": {"containers": [_container("ray-worker", image, worker, WORKER_PORTS)]},
    }
    return {
        "rayVersion": RAY_VERSION,
        "headGroupSpec": {
            "rayStartParams": {"num-cpus": "0", "dashboard-host": "0.0.0.0", "metrics-export-port": "8080"},  # noqa: S104
            "template": {
                "metadata": {"labels": labels},
                "spec": {"containers": [_container("ray-head", image, head, head_ports)]},
            },
        },
        "workerGroupSpecs": [group],
    }


def build_train_rayjob(
    *,
    name: str,
    spec: ModelSpec,
    submission_run_id: str,
    spec_file: str,
    dataset_version: int | None,
    working_dir_uri: str,
    settings: PlatformSettings,
) -> dict[str, Any]:
    """The training RayJob: a cluster for this one job, deleted when it finishes (ADR-0006)."""
    labels = {"modelport.dev/model": spec.name, "modelport.dev/kind": "train"}
    entrypoint = (
        f"python -m modelport_platform.train.entrypoint --model {spec.name}"
        f" --submission-run-id {submission_run_id} --spec-file {spec_file} --rayjob {name}"
    )
    if dataset_version is not None:
        entrypoint += f" --dataset-version {dataset_version}"
    runtime_env: dict[str, Any] = {"working_dir": working_dir_uri}
    if spec.runtime.pip:
        runtime_env["pip"] = {"packages": list(spec.runtime.pip), "pip_check": False}
    train = spec.resources.train
    return {
        "apiVersion": "ray.io/v1",
        "kind": "RayJob",
        "metadata": {"name": name, "namespace": settings.ray_namespace, "labels": {**_PART_OF, **labels}},
        "spec": {
            "entrypoint": entrypoint,
            "runtimeEnvYAML": yaml.safe_dump(runtime_env, sort_keys=True),
            "shutdownAfterJobFinishes": True,
            "ttlSecondsAfterFinished": 60,
            "activeDeadlineSeconds": 3600,
            "backoffLimit": 1,
            "submissionMode": "K8sJobMode",
            "submitterPodTemplate": {
                "spec": {
                    "restartPolicy": "Never",
                    "containers": [
                        {
                            "name": "ray-job-submitter",
                            "image": settings.ray_image,
                            "imagePullPolicy": "IfNotPresent",
                            "resources": SUBMITTER.resources(),
                        }
                    ],
                }
            },
            "rayClusterSpec": _cluster_spec(
                labels=labels,
                image=settings.ray_image,
                head=head_sizing(settings.profile),
                group_name="workers",
                workers=train.workers,
                worker_cpus=math.ceil(train.cpus_per_worker),
                worker=train_worker_sizing(spec, settings.profile),
                # one logical CPU more than Train takes, so the Ray Data reads feeding it can run
                ray_resources={"CPU": str(math.ceil(train.cpus_per_worker) + 1), TRAIN_SLOT: "1"},
            ),
        },
    }


def serve_workers(profile: Profile) -> int:
    """Serve worker pods: 1 on smoke, 2 on full."""
    return 1 if profile is Profile.SMOKE else 2


def build_rayservice(*, serve_config: Mapping[str, Any], settings: PlatformSettings) -> dict[str, Any]:
    """The one RayService that serves every model; only `serveConfigV2` changes after creation.

    `upgradeStrategy: NewCluster` is explicit so the default-on incremental upgrade never needs a Gateway
    (ADR-0007); KubeRay applies a changed `serveConfigV2` in place.
    """
    labels = {"modelport.dev/kind": "serve"}
    workers = serve_workers(settings.profile)
    return {
        "apiVersion": "ray.io/v1",
        "kind": "RayService",
        "metadata": {
            "name": settings.rayservice_name,
            "namespace": settings.ray_namespace,
            "labels": {**_PART_OF, **labels},
        },
        "spec": {
            "serviceUnhealthySecondThreshold": 900,
            "deploymentUnhealthySecondThreshold": 300,
            "upgradeStrategy": {"type": "NewCluster"},
            "serveConfigV2": yaml.safe_dump(dict(serve_config), sort_keys=True),
            "rayClusterConfig": _cluster_spec(
                labels=labels,
                image=settings.ray_image,
                head=head_sizing(settings.profile),
                group_name="serve",
                workers=workers,
                worker_cpus=2,
                worker=serve_worker_sizing(settings.profile),
                head_ports=SERVE_HEAD_PORTS,
            ),
        },
    }
