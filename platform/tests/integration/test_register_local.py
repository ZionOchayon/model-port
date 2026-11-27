"""register_version against a real MLflow (sqlite, local artifacts): lineage, idempotency, artifacts."""

import json
import sys
import textwrap

import mlflow
import numpy as np
import pandas as pd
import pytest

from modelport.lifecycle import ModelState
from modelport.naming import Tag
from modelport.registry import MlflowRegistry
from modelport.spec import Framework, load_spec
from modelport_platform.train.register import Lineage, register_version

pytestmark = pytest.mark.integration

SPEC = """\
name: tiny-tabular
class: tiny_model:Tiny
framework: xgboost
labels: [low, high]
signature:
  inputs:
    - {name: x, type: double}
runtime:
  pip: [xgboost==3.4.1]
data:
  dataset: s3://datasets/tiny
"""
MODEL = """
import json
from pathlib import Path

import numpy as np

from modelport import BaseModel, Predictions


class Tiny(BaseModel):
    def load(self, artifacts_dir):
        self.threshold = json.loads((Path(artifacts_dir) / "w.json").read_text())["t"]

    def preprocess(self, batch):
        return batch["x"].to_numpy()

    def predict(self, inputs):
        labels = ["high" if v > self.threshold else "low" for v in inputs]
        return Predictions(labels, np.where(inputs > self.threshold, inputs, 1 - inputs))

    def train_loop(self, config):
        pass
"""


@pytest.fixture
def setup(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    model_dir = tmp_path / "model"
    (model_dir / "example").mkdir(parents=True)
    (model_dir / "model.tiny.yaml").write_text(SPEC, encoding="utf-8")
    (model_dir / "tiny_model.py").write_text(textwrap.dedent(MODEL), encoding="utf-8")
    val = pd.DataFrame({"x": np.random.default_rng(0).random(800)})
    val.head(5).to_parquet(model_dir / "example" / "input.parquet")
    (tmp_path / "weights").mkdir()
    (tmp_path / "weights" / "w.json").write_text('{"t": 0.5}', encoding="utf-8")
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    mlflow.set_tracking_uri(uri)
    registry = MlflowRegistry(uri, sleep=lambda s: None)
    run_id = registry.ensure_run(
        "modelport-train/tiny-tabular",
        "train-tiny-1",
        Tag.TRAIN_RAYJOB,
        {Tag.TRAIN_RAYJOB: "train-tiny-1"},
        {},
    )
    lineage = Lineage(
        dataset="tiny",
        dataset_version=2,
        dataset_sha256="d" * 64,
        val_uri="s3://datasets/tiny/v2/val.parquet",
        code_sha256="c" * 64,
        spec_sha256="s" * 64,
        submission_run_id="sub-1",
        train_run_id=run_id,
        train_rayjob="train-tiny-1",
        image="modelport/ray:test",
        runtime_pip=("xgboost==3.4.1",),
        framework=Framework.XGBOOST,
    )
    kwargs = {
        "spec": load_spec(model_dir / "model.tiny.yaml"),
        "spec_path": model_dir / "model.tiny.yaml",
        "model_dir": model_dir,
        "weights_dir": tmp_path / "weights",
        "val": val,
        "example": pd.read_parquet(model_dir / "example" / "input.parquet"),
        "lineage": lineage,
        "registry": registry,
    }
    yield kwargs, registry
    sys.modules.pop("tiny_model", None)
    mlflow.set_tracking_uri(None)


def test_register_sets_all_lineage_tags(setup):
    kwargs, _registry = setup
    info = register_version(**kwargs)
    lineage = kwargs["lineage"]
    assert info.version == "1"
    assert info.state is ModelState.REGISTERED
    for key, value in lineage.to_tags().items():
        assert info.tags[key] == value
    assert set(lineage.to_tags()) == {
        Tag.DATASET,
        Tag.DATASET_VERSION,
        Tag.DATASET_SHA256,
        Tag.VAL_URI,
        Tag.CODE_SHA256,
        Tag.SPEC_SHA256,
        Tag.SUBMISSION_RUN_ID,
        Tag.TRAIN_RUN_ID,
        Tag.TRAIN_RAYJOB,
        Tag.IMAGE,
        Tag.RUNTIME_PIP,
        Tag.FRAMEWORK,
    }
    assert json.loads(info.tags[Tag.RUNTIME_PIP]) == ["xgboost==3.4.1"]
    assert info.tags[Tag.DATASET_VERSION] == "2"


def test_register_twice_returns_same_version(setup):
    kwargs, registry = setup
    first = register_version(**kwargs)
    again = register_version(**kwargs)
    assert first.version == again.version
    assert len(registry.client.search_model_versions("name = 'tiny-tabular'")) == 1


def test_baseline_and_spec_artifacts_logged(setup):
    kwargs, registry = setup
    register_version(**kwargs)
    run_id = kwargs["lineage"].train_run_id
    files = {a.path for a in registry.client.list_artifacts(run_id)}
    assert {"spec", "example", "train", "drift"} <= files
    assert {a.path for a in registry.client.list_artifacts(run_id, "spec")} == {"spec/model.yaml"}
    assert {a.path for a in registry.client.list_artifacts(run_id, "example")} == {"example/input.parquet"}
    assert registry.read_json(run_id, "train/runtime_pip.json") == {"pip": ["xgboost==3.4.1"]}
    baseline = registry.read_json(run_id, "drift/baseline.json")
    assert (baseline["model"], baseline["version"], baseline["n_rows"]) == ("tiny-tabular", "1", 800)
    assert set(baseline["features"]) == {"stat_x", "confidence", "prediction"}


def test_signature_on_version(setup):
    kwargs, _ = setup
    info = register_version(**kwargs)
    signature = mlflow.models.get_model_info(f"models:/tiny-tabular/{info.version}").signature
    assert [c.name for c in signature.inputs.inputs] == ["x"]
    assert [c.name for c in signature.outputs.inputs] == ["prediction", "confidence"]
