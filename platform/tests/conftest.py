"""Shared platform test setup."""

import os
import sys
from pathlib import Path

import pytest

# Ray reads this at import time. Under `uv run`, Ray would otherwise start every worker through
# `uv run` again and re-resolve the whole project; local tests want the driver's own interpreter.
os.environ.setdefault("RAY_ENABLE_UV_RUN_RUNTIME_ENV", "0")
sys.path.insert(0, str(Path(__file__).parent))  # `import fakes`

PLATFORM_ENV = {
    "MODELPORT_PROFILE": "smoke",
    "MODELPORT_S3_ENDPOINT": "http://minio.platform.svc.cluster.local:9000",
    "AWS_ACCESS_KEY_ID": "test-key",
    "AWS_SECRET_ACCESS_KEY": "test-secret",
    "MLFLOW_TRACKING_URI": "http://mlflow.platform.svc.cluster.local:5000",
    "MODELPORT_RAY_IMAGE": "modelport/ray:test",
}


@pytest.fixture
def platform_settings():
    """PlatformSettings as the smoke profile sets them in the cluster."""
    from modelport_platform.settings import PlatformSettings

    return PlatformSettings.from_env(PLATFORM_ENV)


@pytest.fixture
def store():
    """An empty in-memory DatasetStore."""
    from fakes import InMemoryDatasetStore

    return InMemoryDatasetStore()
