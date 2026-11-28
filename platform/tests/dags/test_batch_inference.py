import importlib.util
import sys
from datetime import date
from pathlib import Path

import pytest

from modelport.naming import Alias, Tag

pytest.importorskip("airflow.sdk")

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "dags"))
import batch_inference as dag_module

ROOT = Path(__file__).resolve().parents[3]
SPECS = {
    "covertype-classifier": (ROOT / "examples/models/covertype-classifier/model.yaml").read_bytes(),
    "fashion-classifier": (ROOT / "examples/models/image-classifier/model.fashion.yaml").read_bytes(),
}
_spec = importlib.util.spec_from_file_location("sdk_fakes", ROOT / "sdk/tests/fakes.py")
sdk_fakes = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sdk_fakes)
DAY = date(2026, 1, 1)


def registry():
    """covertype: champion v2 of 2; fashion: champion v1; unserved: no champion."""
    reg = sdk_fakes.InMemoryRegistry()
    for model, versions in (("covertype-classifier", 2), ("fashion-classifier", 1), ("unserved", 1)):
        reg.ensure_registered_model(model, {})
        for n in range(1, versions + 1):
            run = f"{model}-train-{n}"
            spec = SPECS.get(model, SPECS["covertype-classifier"])
            reg.runs[run] = {"tags": {}, "artifacts": {"spec/model.yaml": spec}}
            reg.add_version(model, {Tag.TRAIN_RUN_ID: run, Tag.RUNTIME_PIP: f'["{model}-pip=={n}"]'})
        if model != "unserved":
            reg.aliases[model][Alias.CHAMPION] = str(versions)
    return reg


def inputs(store, *datasets):
    for dataset in datasets:
        store.write_bytes(f"s3://batch-input/{dataset}/{DAY.isoformat()}/part-0.parquet", b"x")


def test_plan_locks_version(store):
    reg = registry()
    inputs(store, "covertype", "fashion-mnist")
    planned = dag_module.plan(reg, store, None, DAY)
    by_model = {p["model"]: p for p in planned}
    assert set(by_model) == {"covertype-classifier", "fashion-classifier"}  # no champion, no batch
    cov = by_model["covertype-classifier"]
    assert cov["version"] == "2"
    assert cov["input_uri"] == "s3://batch-input/covertype/2026-01-01/"
    assert cov["output_uri"] == "s3://predictions/covertype-classifier/2026-01-01/"
    assert cov["runtime_pip"] == ["covertype-classifier-pip==2"]
    reg.aliases["covertype-classifier"][Alias.CHAMPION] = "1"  # a promotion after plan
    item = dag_module.batch_job(cov, force=False)
    assert "--version 2" in item["entrypoint"]  # the job uses the version plan locked
    only = dag_module.plan(reg, store, ["fashion-classifier"], DAY)
    assert [p["model"] for p in only] == ["fashion-classifier"]


def test_plan_skips_empty_input(store):
    inputs(store, "covertype")
    planned = dag_module.plan(registry(), store, None, DAY)
    assert [p["model"] for p in planned] == ["covertype-classifier"]


def test_submission_id_contains_day_and_version(store):
    inputs(store, "covertype")
    (cov,) = dag_module.plan(registry(), store, None, DAY)
    job = dag_module.batch_job(cov, force=True)
    assert job["submission_id"] == "batch__covertype-classifier__2026-01-01-v2"
    assert job["force"] is True
    assert job["runtime_env"]["pip"]["packages"] == ["covertype-classifier-pip==2"]
    assert f"--batch-size {cov['batch_size']}" in job["entrypoint"]
