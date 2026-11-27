"""The evaluation job against a real MLflow (sqlite): covertype example registered as v1."""

import json
import shutil
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import pytest
import xgboost

from modelport.pyfunc import log_version
from modelport.registry import MlflowRegistry
from modelport.spec import load_spec
from modelport_platform.gate import evaluate

pytestmark = pytest.mark.integration
MODEL_DIR = Path(__file__).resolve().parents[3] / "examples" / "models" / "covertype-classifier"
VAL_URI = "s3://datasets/covertype/v1/val.parquet"


def frame(spec, n=400, seed=0):
    rng = np.random.default_rng(seed)
    data = {}
    for column in spec.signature.inputs:
        data[column.name] = rng.integers(0, 4, n) if column.type == "long" else rng.random(n)
    data[spec.data.label_column] = rng.choice(spec.labels, n)
    return pd.DataFrame(data)


@pytest.fixture
def registered(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    model_dir = tmp_path / "covertype-classifier"
    shutil.copytree(MODEL_DIR, model_dir)
    spec = load_spec(model_dir / "model.yaml")
    val = frame(spec)
    features = [c.name for c in spec.signature.inputs]
    labels = val[spec.data.label_column].map({label: i for i, label in enumerate(spec.labels)})
    booster = xgboost.train(
        {"objective": "multi:softprob", "num_class": len(spec.labels)},
        xgboost.DMatrix(val[features].astype(np.float32), label=labels),
        num_boost_round=2,
    )
    weights = tmp_path / "weights"
    weights.mkdir()
    booster.save_model(str(weights / "model.ubj"))
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    mlflow.set_tracking_uri(uri)
    with mlflow.start_run():
        logged = log_version(
            spec=spec,
            spec_path=model_dir / "model.yaml",
            model_dir=model_dir,
            weights_dir=weights,
            example=val[features].head(5),
            registered_model_name=spec.name,
        )
    registry = MlflowRegistry(uri, sleep=lambda s: None)
    report_run = registry.client.create_run(registry.client.create_experiment("modelport-eval")).info.run_id
    yield {"spec": spec, "val": val, "uri": logged.model_uri, "registry": registry, "report_run": report_run}
    mlflow.set_tracking_uri(None)


def args(ctx):
    return evaluate.parse_args(
        [
            "--model",
            ctx["spec"].name,
            "--version",
            "1",
            "--val-uri",
            VAL_URI,
            "--report-run-id",
            ctx["report_run"],
        ]
    )


def test_report_fields_complete(registered):
    ctx = registered
    code = evaluate.run(args(ctx), registry=ctx["registry"], read_val=lambda uri: ctx["val"])
    assert code == 0
    report = ctx["registry"].read_json(ctx["report_run"], "gate/report.json")
    assert set(report) == {
        "model",
        "version",
        "val_uri",
        "n_samples",
        "metric_name",
        "metric",
        "per_class",
        "latency_ms_median",
        "latency_batch_size",
        "created_at",
    }
    assert (report["model"], report["version"], report["val_uri"], report["n_samples"]) == (
        ctx["spec"].name,
        "1",
        VAL_URI,
        400,
    )
    assert set(report["per_class"]) == set(ctx["spec"].labels)
    assert 0.0 <= report["metric"] <= 1.0
    assert report["latency_ms_median"] > 0
    assert report["created_at"].endswith("Z")
    metrics = ctx["registry"].client.get_run(ctx["report_run"]).data.metrics
    assert set(metrics) >= {"eval_metric", "eval_latency_ms_median"}
    json.dumps(report)


def test_latency_repeats_used(registered, monkeypatch):
    ctx = registered
    calls = []
    original = evaluate.measure_latency

    def spy(model, frame, *, batch_size, repeats, **kw):
        calls.append((batch_size, repeats))
        return original(model, frame, batch_size=batch_size, repeats=repeats, **kw)

    monkeypatch.setattr(evaluate, "measure_latency", spy)
    assert evaluate.run(args(ctx), registry=ctx["registry"], read_val=lambda uri: ctx["val"]) == 0
    spec = ctx["spec"]
    assert calls == [(spec.evaluation.latency_batch_size, spec.evaluation.latency_repeats)]


def test_label_column_missing_exit_4(registered, capsys):
    ctx = registered
    val = ctx["val"].drop(columns=[ctx["spec"].data.label_column])
    assert evaluate.run(args(ctx), registry=ctx["registry"], read_val=lambda uri: val) == 4
    assert "cover_type" in capsys.readouterr().err
