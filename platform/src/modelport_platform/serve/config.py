"""The complete RayService `serveConfigV2` from the desired state of every served model (pure core).

Each model gets a router app at `/<model>` and one app per served version (`<model>--champion`, and
`<model>--candidate` while a canary runs), each with its own `runtime_env`. The whole state is rendered
every time, sorted by model name, so the same state always renders the same config and applying it is
idempotent.
"""

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final

from modelport.naming import Tag
from modelport.spec import ModelSpec

if TYPE_CHECKING:
    from modelport.registry import VersionInfo

HTTP_PORT: Final = 8000
ROUTER_CPUS: Final = 0.1
_UNITS: Final = {"Mi": 2**20, "Gi": 2**30}


def parse_memory(quantity: str) -> int:
    """`512Mi` or `1Gi` (the units `model.yaml` allows) in bytes.

    Raises:
        ValueError: any other format.
    """
    match = re.fullmatch(r"([1-9][0-9]*)(Mi|Gi)", quantity)
    if match is None:
        raise ValueError(f"memory quantity {quantity!r} must be <n>Mi or <n>Gi")
    return int(match.group(1)) * _UNITS[match.group(2)]


@dataclass(frozen=True)
class ServedVariant:
    """One served version and its replica sizing."""

    version: str
    runtime_pip: tuple[str, ...]
    replicas: int
    num_cpus: float
    memory_bytes: int
    fault_error_rate: float = 0.0


@dataclass(frozen=True)
class ServedModel:
    """What one model should serve: its champion and, during a canary, a candidate with a traffic weight."""

    name: str
    champion: ServedVariant | None  # None for a model's first version, served as the candidate only
    candidate: ServedVariant | None
    candidate_weight: float
    latency_threshold_ms: float
    live_confidence_drop: float
    live_class_share_increase: float
    inference_log_sample_rate: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.candidate_weight <= 1.0:
            raise ValueError(f"candidate_weight {self.candidate_weight} must be between 0 and 1")
        if self.champion is None and self.candidate is None:
            raise ValueError(f"{self.name} has neither a champion nor a candidate to serve")


def _variant(spec: ModelSpec, info: "VersionInfo", fault_error_rate: float) -> ServedVariant:
    serve = spec.resources.serve
    return ServedVariant(
        version=info.version,
        runtime_pip=tuple(json.loads(info.tags.get(Tag.RUNTIME_PIP, "[]"))),
        replicas=serve.replicas,
        num_cpus=serve.cpus,
        memory_bytes=parse_memory(serve.memory),
        fault_error_rate=fault_error_rate,
    )


def served_model_from_spec(
    spec: ModelSpec,
    champion: "VersionInfo | None",
    candidate: "VersionInfo | None",
    *,
    candidate_weight: float,
    fault_error_rate: float = 0.0,
) -> ServedModel:
    """The desired state of one model; `fault_error_rate` (the canary drill) applies to the candidate only.

    Without a champion (a model's first version) all traffic goes to the candidate.
    """
    if champion is None:
        candidate_weight = 1.0
    return ServedModel(
        name=spec.name,
        champion=_variant(spec, champion, 0.0) if champion is not None else None,
        candidate=_variant(spec, candidate, fault_error_rate) if candidate is not None else None,
        candidate_weight=candidate_weight if candidate is not None else 0.0,
        latency_threshold_ms=spec.serving.latency_p95_ms,
        live_confidence_drop=spec.drift.live_confidence_drop,
        live_class_share_increase=spec.drift.live_class_share_increase,
        inference_log_sample_rate=spec.drift.inference_log_sample_rate,
    )


def _version_app(
    model: ServedModel, variant: ServedVariant, role: str, env_vars: Mapping[str, str]
) -> dict[str, Any]:
    runtime_env: dict[str, Any] = {}
    if variant.runtime_pip:
        runtime_env["pip"] = {"packages": list(variant.runtime_pip), "pip_check": False}
    runtime_env["env_vars"] = dict(sorted(env_vars.items()))
    return {
        "name": f"{model.name}--{role}",
        "route_prefix": None,  # reachable only through the router
        "import_path": "modelport_platform.serve.app:build_model_app",
        "runtime_env": runtime_env,
        "args": {"model": model.name, "version": variant.version, "variant": role},
        "deployments": [
            {
                "name": "ModelDeployment",
                "num_replicas": variant.replicas,
                "ray_actor_options": {"num_cpus": variant.num_cpus, "memory": variant.memory_bytes},
                "user_config": {
                    "fault_error_rate": variant.fault_error_rate,
                    "inference_log_sample_rate": model.inference_log_sample_rate,
                    "live_confidence_drop": model.live_confidence_drop,
                    "live_class_share_increase": model.live_class_share_increase,
                },
            }
        ],
    }


def render_serve_config(models: Sequence[ServedModel], *, env_vars: Mapping[str, str]) -> dict[str, Any]:
    """`serveConfigV2` for every model, sorted by name."""
    applications: list[dict[str, Any]] = []
    for model in sorted(models, key=lambda m: m.name):
        champion, candidate = model.champion, model.candidate
        weight = 1.0 if champion is None else (model.candidate_weight if candidate else 0.0)
        applications.append(
            {
                "name": model.name,
                "route_prefix": f"/{model.name}",
                "import_path": "modelport_platform.serve.router:build_router_app",
                "args": {"model": model.name},
                "deployments": [
                    {
                        "name": "ModelRouter",
                        "num_replicas": 1,
                        "ray_actor_options": {"num_cpus": ROUTER_CPUS},
                        "user_config": {
                            "champion_app": f"{model.name}--champion" if champion else None,
                            "candidate_app": f"{model.name}--candidate" if candidate else None,
                            "candidate_weight": weight,
                            "latency_threshold_ms": model.latency_threshold_ms,
                        },
                    }
                ],
            }
        )
        if champion is not None:
            applications.append(_version_app(model, champion, "champion", env_vars))
        if candidate is not None:
            applications.append(_version_app(model, candidate, "candidate", env_vars))
    return {"http_options": {"host": "0.0.0.0", "port": HTTP_PORT}, "applications": applications}  # noqa: S104
