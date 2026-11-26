import pytest
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import RESOURCE_DOES_NOT_EXIST

from modelport.errors import NotFoundError, PlatformUnavailableError
from modelport.registry import MlflowRegistry


class DownClient:
    calls = 0

    def get_registered_model(self, name):
        DownClient.calls += 1
        raise ConnectionError("connection refused")


class MissingClient:
    def get_registered_model(self, name):
        raise MlflowException("no such model", error_code=RESOURCE_DOES_NOT_EXIST)


def test_unreachable_raises_platform_unavailable_after_retries():
    sleeps = []
    registry = MlflowRegistry("http://mlflow.invalid:5000", client=DownClient(), sleep=sleeps.append)
    with pytest.raises(PlatformUnavailableError) as info:
        registry.snapshot("image-classifier")
    assert DownClient.calls == 4
    assert sleeps == [1, 2, 4]
    assert "http://mlflow.invalid:5000" in info.value.what
    assert "make ui" in info.value.fix


def test_unknown_model_snapshot_raises_not_found():
    registry = MlflowRegistry("http://x", client=MissingClient(), sleep=lambda s: None)
    with pytest.raises(NotFoundError) as info:
        registry.snapshot("image-classifier")
    assert info.value.what == "No model named 'image-classifier' is registered"
    assert "modelport status" in info.value.fix


def test_logical_errors_are_not_retried():
    sleeps = []
    registry = MlflowRegistry("http://x", client=MissingClient(), sleep=sleeps.append)
    with pytest.raises(NotFoundError):
        registry.snapshot("image-classifier")
    assert sleeps == []


def test_artifact_s3_uri_maps_proxied_artifacts():
    class RunClient:
        def get_run(self, run_id):
            class Info:
                artifact_uri = "mlflow-artifacts:/3/abc/artifacts"

            class Run:
                info = Info()

            return Run()

    registry = MlflowRegistry("http://x", client=RunClient(), sleep=lambda s: None)
    assert registry.artifact_s3_uri("abc", "code/code.zip") == "s3://mlflow/3/abc/artifacts/code/code.zip"
