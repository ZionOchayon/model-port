"""The batch job on local Ray against a real MLflow (sqlite) and a local filesystem."""

import json
import shutil
from datetime import date
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import pytest
import xgboost
from pyarrow import fs

from modelport.pyfunc import log_version
from modelport.spec import load_spec

pytestmark = pytest.mark.integration
MODEL_DIR = Path(__file__).resolve().parents[3] / "examples" / "models" / "covertype-classifier"
DAY = date(2026, 1, 1)


def features(spec, n, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {c.name: rng.integers(0, 4, n) if c.type == "long" else rng.random(n) for c in spec.signature.inputs}
    )


@pytest.fixture
def batch(tmp_path, monkeypatch, local_ray):
    from modelport_platform.batch import job

    monkeypatch.chdir(tmp_path)
    model_dir = tmp_path / "covertype-classifier"
    shutil.copytree(MODEL_DIR, model_dir)
    spec = load_spec(model_dir / "model.yaml")
    train = features(spec, 200)
    labels = np.random.default_rng(1).integers(0, len(spec.labels), len(train))
    booster = xgboost.train(
        {"objective": "multi:softprob", "num_class": len(spec.labels)},
        xgboost.DMatrix(train.astype(np.float32), label=labels),
        num_boost_round=2,
    )
    weights = tmp_path / "weights"
    weights.mkdir()
    booster.save_model(str(weights / "model.ubj"))
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    mlflow.set_tracking_uri(uri)
    with mlflow.start_run():
        log_version(
            spec=spec,
            spec_path=model_dir / "model.yaml",
            model_dir=model_dir,
            weights_dir=weights,
            example=train.head(5),
            registered_model_name=spec.name,
        )
    input_dir = tmp_path / "batch-input"
    input_dir.mkdir()
    features(spec, 300, seed=2).to_parquet(input_dir / "part-0.parquet")
    output_dir = tmp_path / "predictions"

    def run():
        return job.run_batch(
            model=spec.name,
            version="1",
            input_uri=str(input_dir),
            output_uri=str(output_dir),
            day=DAY,
            filesystem=fs.LocalFileSystem(),
            tracking_uri=uri,
            batch_size=64,
            num_cpus=0.5,
        )

    yield {"run": run, "output": output_dir}
    mlflow.set_tracking_uri(None)


def output(path):
    """The written rows in a fixed order (Ray writes files in no particular order)."""
    frame = pd.read_parquet(path)
    return frame.sort_values(list(frame.columns)).reset_index(drop=True)


def test_rerun_same_output_rows(batch):
    assert batch["run"]() == 300
    first = output(batch["output"])
    assert batch["run"]() == 300
    pd.testing.assert_frame_equal(first, output(batch["output"]))


def test_overwrite_no_duplicates(batch):
    batch["run"]()
    batch["run"]()
    assert len(pd.read_parquet(batch["output"])) == 300


def test_success_marker_written(batch):
    batch["run"]()
    marker = json.loads((batch["output"] / "_SUCCESS.json").read_text())
    assert marker["rows"] == 300
    assert (marker["model"], marker["version"], marker["day"]) == ("covertype-classifier", "1", "2026-01-01")


def test_output_has_version_column(batch):
    batch["run"]()
    scored = pd.read_parquet(batch["output"])
    assert {"prediction", "confidence", "model", "version"} <= set(scored.columns)
    assert set(scored["version"]) == {"1"}
    assert set(scored["model"]) == {"covertype-classifier"}
