import importlib.util
import sys
from pathlib import Path

import pytest

from modelport.naming import Tag

pytest.importorskip("airflow.sdk")
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "dags"))
import retrain_on_dataset as dag_module

# The SDK's fakes module has the same name as the platform's; load it under its own name.
_spec = importlib.util.spec_from_file_location(
    "sdk_fakes", Path(__file__).resolve().parents[3] / "sdk/tests/fakes.py"
)
sdk_fakes = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sdk_fakes)
InMemoryRegistry = sdk_fakes.InMemoryRegistry


def registry():
    reg = InMemoryRegistry()
    reg.ensure_registered_model(
        "fashion-classifier",
        {
            Tag.DATASET: "fashion-mnist",
            Tag.AUTO_RETRAIN: "true",
            Tag.LATEST_SUBMISSION_RUN_ID: "run-7",
            Tag.LATEST_SPEC_FILE: "model.fashion.yaml",
        },
    )
    reg.ensure_registered_model("covertype-classifier", {Tag.DATASET: "covertype", Tag.AUTO_RETRAIN: "true"})
    reg.ensure_registered_model(
        "frozen-classifier", {Tag.DATASET: "fashion-mnist", Tag.AUTO_RETRAIN: "false"}
    )
    return reg


def test_subscribers_from_events():
    triggers = dag_module.subscribers([{"dataset": "fashion-mnist", "version": 2, "sha256": "x"}], registry())
    assert triggers == [
        {
            "trigger_run_id": "retrain__fashion-classifier__fashion-mnist__v2",
            "conf": {
                "model": "fashion-classifier",
                "submission_run_id": "run-7",
                "dataset_version": 2,
                "spec_file": "model.fashion.yaml",
            },
        }
    ]


def test_models_without_auto_retrain_skipped():
    names = [
        t["conf"]["model"]
        for t in dag_module.subscribers([{"dataset": "fashion-mnist", "version": 3}], registry())
    ]
    assert names == ["fashion-classifier"]
    assert dag_module.subscribers([{"dataset": "eurosat", "version": 1}], registry()) == []
