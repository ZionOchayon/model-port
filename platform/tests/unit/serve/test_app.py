import asyncio
import io
from datetime import UTC, datetime
from typing import ClassVar

import numpy as np
import pandas as pd
import pytest

from modelport import BaseModel, Predictions
from modelport.naming import Tag
from modelport.registry import VersionInfo
from modelport.spec import parse_spec
from modelport_platform.serve import app

SPEC = parse_spec(
    {
        "name": "tiny",
        "class": "m:Tiny",
        "framework": "xgboost",
        "labels": ["low", "high"],
        "signature": {"inputs": [{"name": "x", "type": "double"}, {"name": "k", "type": "long"}]},
        "data": {"dataset": "s3://datasets/tiny"},
    }
)
GOOD_EXAMPLE = pd.DataFrame({"x": [0.1, 0.9], "k": [1, 2]})
BASELINE = {
    "model": "tiny",
    "version": "3",
    "n_rows": 100,
    "confidence_mean": 0.8,
    "class_shares": {"low": 0.5, "high": 0.5},
    "features": {},
}


class Tiny(BaseModel):
    def load(self, artifacts_dir):
        pass

    def preprocess(self, batch):
        return batch["x"].to_numpy()

    def predict(self, inputs):
        return Predictions(
            labels=np.where(inputs > 0.5, "high", "low").tolist(), confidence=np.full(len(inputs), 0.7)
        )


class FakeRegistry:
    def __init__(self, example):
        self.example = example

    def get_version(self, name, version):
        tags = {Tag.TRAIN_RUN_ID: "train-run"}
        return VersionInfo(name, version, "train-run", "src", tags, (), datetime.now(UTC))

    def model_uri(self, name, version):
        return f"models:/{name}/{version}"

    def download(self, run_id, artifact_path, dst):
        assert (run_id, artifact_path) == ("train-run", "example/input.parquet")
        path = dst / "input.parquet"
        self.example.to_parquet(path)
        return path

    def read_json(self, run_id, artifact_path):
        assert (run_id, artifact_path) == ("train-run", "drift/baseline.json")
        return BASELINE


class Recorder:
    """Stands in for ServeMetrics; records every call."""

    instances: ClassVar[list["Recorder"]] = []

    def __init__(self, model, version):
        self.calls = []
        Recorder.instances.append(self)

    def __getattr__(self, name):
        return lambda *args, **kwargs: self.calls.append((name, args, kwargs))


class Loaded:
    def __init__(self):
        self.model = Tiny(SPEC)


def make(monkeypatch, store, example=GOOD_EXAMPLE):
    Recorder.instances.clear()
    monkeypatch.setattr(app, "_registry", lambda: FakeRegistry(example))
    monkeypatch.setattr(app, "_load_pyfunc", lambda uri: Loaded())
    monkeypatch.setattr(app, "_store", lambda: store)
    monkeypatch.setattr(app, "ServeMetrics", Recorder)
    return app.ModelDeployment.func_or_class("tiny", "3", "champion")


def predict(deployment, body):
    return asyncio.run(deployment.predict(body))


def test_self_test_failure_raises(monkeypatch, store):
    broken = pd.DataFrame({"x": ["not a number"], "k": [1]})
    with pytest.raises(RuntimeError, match="signature self-test failed for tiny v3"):
        make(monkeypatch, store, example=broken)


def test_predict_returns_contract_response(monkeypatch, store):
    deployment = make(monkeypatch, store)
    response = predict(deployment, {"instances": [{"x": 0.9, "k": 1}, {"x": 0.2, "k": 0}]})
    assert response == {
        "model": "tiny",
        "version": "3",
        "variant": "champion",
        "predictions": [{"prediction": "high", "confidence": 0.7}, {"prediction": "low", "confidence": 0.7}],
    }


def test_bad_input_returns_422(monkeypatch, store):
    deployment = make(monkeypatch, store)
    for body in (
        {"instances": [{"x": 0.9}]},  # missing column
        {"instances": [{"x": "high", "k": 1}]},  # wrong type
        {"instances": [{"x": 0.1, "k": 1}] * 1025},  # too many rows
        {"rows": []},  # not a request
    ):
        response = predict(deployment, body)
        assert response["status"] == 422
        assert response["error"]


def test_fault_injection_rate_one_always_raises(monkeypatch, store):
    deployment = make(monkeypatch, store)
    deployment.reconfigure({"fault_error_rate": 1.0})
    with pytest.raises(RuntimeError, match="fault injection"):
        predict(deployment, {"instances": [{"x": 0.9, "k": 1}]})
    deployment.reconfigure({"fault_error_rate": 0.0})
    assert predict(deployment, {"instances": [{"x": 0.9, "k": 1}]})["predictions"]


def test_metrics_observed(monkeypatch, store):
    deployment = make(monkeypatch, store)
    deployment.reconfigure({"live_confidence_drop": 0.2, "live_class_share_increase": 0.4})
    predict(deployment, {"instances": [{"x": 0.9, "k": 1}]})
    calls = {name: (args, kwargs) for name, args, kwargs in Recorder.instances[0].calls}
    assert calls["set_baseline"] == ((0.8, {"low": 0.5, "high": 0.5}), {})
    assert calls["set_model_info"] == (("champion",), {})
    assert calls["set_live_thresholds"] == ((), {"confidence_drop": 0.2, "class_share_increase": 0.4})
    (output, stats), _ = calls["observe"]
    assert output["prediction"].tolist() == ["high"]
    assert list(stats.columns) == ["x", "k"]


def test_inference_log_written_on_flush(monkeypatch, store):
    deployment = make(monkeypatch, store)
    deployment.reconfigure({"inference_log_sample_rate": 1.0})
    predict(deployment, {"instances": [{"x": 0.9, "k": 1}]})
    uri = deployment.log.flush()
    assert uri.startswith("s3://inference-logs/tiny/")
    frame = pd.read_parquet(io.BytesIO(store.objects[uri]))
    assert frame[["prediction", "stat_x", "stat_k"]].to_dict("records") == [
        {"prediction": "high", "stat_x": 0.9, "stat_k": 1.0}
    ]


def test_build_model_app_binds_args():
    application = app.build_model_app({"model": "tiny", "version": "3", "variant": "candidate"})
    assert type(application).__name__ == "Application"


def test_serve_metrics_accept_ray_metric_rules():
    # Ray validates names, tag keys and histogram bounds at construction and on every call
    from modelport_platform.serve.metrics import ServeMetrics

    metrics = ServeMetrics("tiny", "3")
    metrics.set_baseline(0.8, {"low": 0.5, "high": 0.5})
    metrics.set_model_info("champion")
    metrics.set_live_thresholds(confidence_drop=0.15, class_share_increase=0.3)
    metrics.observe(
        pd.DataFrame({"prediction": ["low", "high", "low"], "confidence": [0.6, 0.9, 0.7]}),
        pd.DataFrame({"x": [0.1, 0.2, 0.3], "k": [1.0, 2.0, 3.0]}),
    )
