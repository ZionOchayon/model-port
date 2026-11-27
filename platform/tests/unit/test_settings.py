import pytest

from modelport.errors import ConfigError
from modelport_platform.settings import PlatformSettings, Profile

REQUIRED = {
    "MODELPORT_S3_ENDPOINT": "http://minio.platform.svc.cluster.local:9000",
    "AWS_ACCESS_KEY_ID": "modelport-admin",
    "AWS_SECRET_ACCESS_KEY": "secret",
    "MLFLOW_TRACKING_URI": "http://mlflow.platform.svc.cluster.local:5000",
    "MODELPORT_RAY_IMAGE": "modelport/ray:abc",
}


def test_missing_vars_listed_together():
    with pytest.raises(ConfigError) as info:
        PlatformSettings.from_env({"AWS_ACCESS_KEY_ID": "x"})
    for name in (
        "MODELPORT_S3_ENDPOINT",
        "AWS_SECRET_ACCESS_KEY",
        "MLFLOW_TRACKING_URI",
        "MODELPORT_RAY_IMAGE",
    ):
        assert name in info.value.what
    assert "AWS_ACCESS_KEY_ID" not in info.value.what


def test_defaults():
    s = PlatformSettings.from_env(REQUIRED)
    assert s.profile is Profile.FULL
    assert (s.s3_region, s.ray_namespace, s.rayservice_name) == ("us-east-1", "ray", "modelport-serve")
    assert s.shared_ray_address == "http://modelport-shared-head-svc.ray.svc.cluster.local:8265"
    assert s.serve_url == "http://modelport-serve-serve-svc.ray.svc.cluster.local:8000"
    assert (s.prometheus_url, s.pushgateway_url) == (None, None)
    assert s.canary_synthetic_traffic is True
    assert s.datasets == ("fashion-mnist", "eurosat", "covertype")


def test_profile_parsing():
    assert PlatformSettings.from_env({**REQUIRED, "MODELPORT_PROFILE": "smoke"}).profile is Profile.SMOKE
    with pytest.raises(ConfigError, match="MODELPORT_PROFILE"):
        PlatformSettings.from_env({**REQUIRED, "MODELPORT_PROFILE": "tiny"})


def test_optional_urls_and_lists():
    s = PlatformSettings.from_env(
        {
            **REQUIRED,
            "MODELPORT_PROMETHEUS_URL": "http://kps-prometheus.monitoring.svc.cluster.local:9090",
            "MODELPORT_CANARY_SYNTHETIC_TRAFFIC": "false",
            "MODELPORT_DATASETS": "covertype, fashion-mnist",
        }
    )
    assert s.prometheus_url.endswith(":9090")
    assert s.canary_synthetic_traffic is False
    assert s.datasets == ("covertype", "fashion-mnist")
    assert "secret" not in repr(s)
