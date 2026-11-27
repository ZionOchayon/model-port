"""Platform configuration from the environment (twelve-factor).

In the cluster it comes from the platform ConfigMap and Secrets, so nothing is baked into images.
"""

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Self

from modelport.errors import ConfigError

REQUIRED = (
    "MODELPORT_S3_ENDPOINT",
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "MLFLOW_TRACKING_URI",
    "MODELPORT_RAY_IMAGE",
)


class Profile(StrEnum):
    """`make up` profile: full (16 GB) or smoke (CI runner)."""

    SMOKE = "smoke"
    FULL = "full"


@dataclass(frozen=True)
class PlatformSettings:
    """Where every platform service is and how the platform behaves."""

    profile: Profile
    s3_endpoint: str
    s3_access_key: str
    s3_secret_key: str = field(repr=False)
    s3_region: str
    mlflow_uri: str
    ray_namespace: str
    ray_image: str
    shared_ray_address: str
    rayservice_name: str
    serve_url: str
    prometheus_url: str | None
    pushgateway_url: str | None
    canary_synthetic_traffic: bool
    datasets: tuple[str, ...]

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> Self:
        """Read the settings (pure parsing of environment variables).

        Raises:
            ConfigError: listing every missing variable at once, or an invalid profile.
        """
        e = os.environ if env is None else env
        missing = [name for name in REQUIRED if not e.get(name)]
        if missing:
            raise ConfigError(
                what=f"Missing platform settings: {', '.join(missing)}",
                fix="These come from the modelport-platform-env ConfigMap and the S3 Secret; run `make up`.",
                details={"missing": missing},
            )
        try:
            profile = Profile(e.get("MODELPORT_PROFILE", Profile.FULL))
        except ValueError:
            raise ConfigError(
                what=f"MODELPORT_PROFILE is '{e['MODELPORT_PROFILE']}'", fix="Use smoke or full."
            ) from None
        return cls(
            profile=profile,
            s3_endpoint=e["MODELPORT_S3_ENDPOINT"],
            s3_access_key=e["AWS_ACCESS_KEY_ID"],
            s3_secret_key=e["AWS_SECRET_ACCESS_KEY"],
            s3_region=e.get("AWS_REGION", "us-east-1"),
            mlflow_uri=e["MLFLOW_TRACKING_URI"],
            ray_namespace=e.get("MODELPORT_RAY_NAMESPACE", "ray"),
            ray_image=e["MODELPORT_RAY_IMAGE"],
            shared_ray_address=e.get(
                "MODELPORT_SHARED_RAY_ADDRESS", "http://modelport-shared-head-svc.ray.svc.cluster.local:8265"
            ),
            rayservice_name=e.get("MODELPORT_RAYSERVICE_NAME", "modelport-serve"),
            serve_url=e.get(
                "MODELPORT_SERVE_URL", "http://modelport-serve-serve-svc.ray.svc.cluster.local:8000"
            ),
            prometheus_url=e.get("MODELPORT_PROMETHEUS_URL") or None,
            pushgateway_url=e.get("MODELPORT_PUSHGATEWAY_URL") or None,
            canary_synthetic_traffic=e.get("MODELPORT_CANARY_SYNTHETIC_TRAFFIC", "true").lower() == "true",
            datasets=tuple(
                d.strip()
                for d in e.get("MODELPORT_DATASETS", "fashion-mnist,eurosat,covertype").split(",")
                if d.strip()
            ),
        )
