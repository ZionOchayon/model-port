"""Log a BaseModel as an MLflow pyfunc and load it back (sqlite MLflow, local artifacts)."""

import os
import shutil
import socket
import subprocess
import sys
import textwrap
import time
import urllib.request
from pathlib import Path

import mlflow
import pandas as pd
import pytest

from modelport.errors import NotFoundError
from modelport.pyfunc import ModelportPyfunc, load_version, log_version
from modelport.spec import load_spec

pytestmark = pytest.mark.integration

SDK = Path(__file__).resolve().parents[2]
SPEC = """\
name: tiny-tabular
class: tiny_model:Tiny
framework: xgboost
labels: [low, high]
signature:
  inputs:
    - {name: x, type: double}
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
        weights = Path(artifacts_dir) / "w.json" if artifacts_dir else None
        self.threshold = json.loads(weights.read_text())["t"] if weights else 0.5

    def preprocess(self, batch):
        return batch["x"].to_numpy()

    def predict(self, inputs):
        labels = ["high" if v > self.threshold else "low" for v in inputs]
        return Predictions(labels, np.full(len(inputs), 0.9))

    def train_loop(self, config):
        pass
"""
EXAMPLE = pd.DataFrame({"x": [0.1, 0.4, 0.6, 0.9]})


def write_model(root: Path) -> Path:
    model_dir = root / "tiny-model"
    model_dir.mkdir(parents=True)
    (model_dir / "model.yaml").write_text(SPEC, encoding="utf-8")
    (model_dir / "tiny_model.py").write_text(textwrap.dedent(MODEL), encoding="utf-8")
    (root / "weights").mkdir()
    (root / "weights" / "w.json").write_text('{"t": 0.5}', encoding="utf-8")
    return model_dir


@pytest.fixture
def logged(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    model_dir = write_model(tmp_path)
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")
    spec = load_spec(model_dir / "model.yaml")
    with mlflow.start_run() as run:
        version = log_version(
            spec=spec,
            spec_path=model_dir / "model.yaml",
            model_dir=model_dir,
            weights_dir=tmp_path / "weights",
            example=EXAMPLE,
            registered_model_name=spec.name,
        )
    sys.modules.pop("tiny_model", None)
    yield version, run.info.run_id, spec, model_dir
    mlflow.set_tracking_uri(None)


def test_log_and_load_roundtrip(logged):
    version, run_id, spec, _ = logged
    assert (version.version, version.run_id, version.model_uri) == ("1", run_id, "models:/tiny-tabular/1")
    loaded = load_version(version.model_uri)
    assert isinstance(loaded, ModelportPyfunc)
    assert loaded.predict(None, EXAMPLE)["prediction"].tolist() == ["low", "low", "high", "high"]
    assert loaded.spec == spec
    enforced = mlflow.pyfunc.load_model(version.model_uri).predict(EXAMPLE.head(2))
    assert enforced["prediction"].tolist() == ["low", "low"]


def test_signature_recorded(logged):
    version, *_ = logged
    signature = mlflow.models.get_model_info(version.model_uri).signature
    assert [c.name for c in signature.inputs.inputs] == ["x"]
    assert [c.name for c in signature.outputs.inputs] == ["prediction", "confidence"]


def test_code_paths_restore_user_module(logged, tmp_path):
    version, *_, model_dir = logged
    shutil.rmtree(model_dir)  # only the copy inside the MLflow model may be used
    sys.modules.pop("tiny_model", None)
    loaded = load_version(version.model_uri)
    assert type(loaded.model).__name__ == "Tiny"
    assert not Path(sys.modules[type(loaded.model).__module__].__file__).resolve().is_relative_to(model_dir)


def test_missing_version_not_found(logged):
    with pytest.raises(NotFoundError):
        load_version("models:/tiny-tabular/99")


SKINNY = """
import importlib.util
import sys
from pathlib import Path

import mlflow
import pandas as pd

from modelport.pyfunc import load_version, log_version
from modelport.spec import load_spec

root, uri = Path(sys.argv[1]), sys.argv[2]
mlflow.set_tracking_uri(uri)
spec = load_spec(root / "tiny-model" / "model.yaml")
frame = pd.DataFrame({"x": [0.2, 0.8]})
with mlflow.start_run():
    v = log_version(
        spec=spec,
        spec_path=root / "tiny-model" / "model.yaml",
        model_dir=root / "tiny-model",
        weights_dir=root / "weights",
        example=frame,
        registered_model_name=spec.name,
    )
print(load_version(v.model_uri).predict(None, frame)["prediction"].tolist())
assert importlib.util.find_spec("sqlalchemy") is None, "full mlflow leaked into the skinny environment"
"""


@pytest.fixture
def mlflow_server(tmp_path):
    """A real MLflow server (sqlite, proxied local artifacts), like the in-cluster one but on localhost."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    args = [
        sys.executable,
        "-m",
        "mlflow",
        "server",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--workers",
        "1",
    ]
    args += ["--backend-store-uri", f"sqlite:///{tmp_path / 'server.db'}"]
    args += ["--artifacts-destination", str(tmp_path / "artifacts")]
    env = {**os.environ, "MLFLOW_SERVER_ENABLE_JOB_EXECUTION": "false"}
    server = subprocess.Popen(args, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    url = f"http://127.0.0.1:{port}"
    try:
        for _ in range(120):
            try:
                urllib.request.urlopen(f"{url}/health", timeout=1)
                break
            except OSError:
                time.sleep(0.5)
        else:
            pytest.fail("MLflow server did not start")
        yield url
    finally:
        server.terminate()
        server.wait(timeout=30)


def test_skinny_is_enough(tmp_path, mlflow_server):
    """Assumption A6: mlflow-skinny + pandas/numpy/pyarrow log and load the pyfunc through the server."""
    uv = shutil.which("uv")
    assert uv is not None, "uv builds the isolated skinny environment"
    work = tmp_path / "work"
    write_model(work)
    (work / "check.py").write_text(SKINNY, encoding="utf-8")
    env = {k: v for k, v in os.environ.items() if not k.startswith(("VIRTUAL_ENV", "UV_PROJECT"))}
    command = [uv, "run", "--isolated", "--no-project", "--python", "3.12", "--with", str(SDK), "--"]
    command += ["python", str(work / "check.py"), str(work), mlflow_server]
    result = subprocess.run(
        command, capture_output=True, text=True, check=False, env=env, timeout=600, cwd=work
    )
    assert result.returncode == 0, result.stderr[-3000:]
    assert "['low', 'high']" in result.stdout
