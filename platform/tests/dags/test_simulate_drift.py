import importlib.util
import io
import sys
from pathlib import Path

import pandas as pd
import pytest

from modelport.naming import Alias, Tag

pytest.importorskip("airflow.sdk")
from airflow.sdk.exceptions import AirflowFailException, ParamValidationError

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "dags"))
import simulate_drift as dag_module

ROOT = Path(__file__).resolve().parents[3]
SPECS = {
    "covertype-classifier": (ROOT / "examples/models/covertype-classifier/model.yaml").read_bytes(),
    "fashion-classifier": (ROOT / "examples/models/image-classifier/model.fashion.yaml").read_bytes(),
}
_spec = importlib.util.spec_from_file_location("sdk_fakes", ROOT / "sdk/tests/fakes.py")
sdk_fakes = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sdk_fakes)


def registry(store):
    reg = sdk_fakes.InMemoryRegistry()
    for model, spec in SPECS.items():
        reg.ensure_registered_model(model, {})
        run = f"{model}-train-1"
        reg.runs[run] = {"tags": {}, "artifacts": {"spec/model.yaml": spec}}
        val = f"s3://datasets/{model}/v1/val.parquet"
        buffer = io.BytesIO()
        pd.DataFrame({"x": range(10)}).to_parquet(buffer)
        store.write_bytes(val, buffer.getvalue())
        reg.add_version(model, {Tag.TRAIN_RUN_ID: run, Tag.VAL_URI: val})
        reg.aliases[model][Alias.CHAMPION] = "1"
    return reg


def test_perturbation_model_kind_mismatch_fails(store):
    reg = registry(store)
    with pytest.raises(AirflowFailException, match="shift_feature"):
        dag_module.load_rows(reg, store, "fashion-classifier", "shift_feature", None, rows=5)
    with pytest.raises(AirflowFailException, match="image perturbation"):
        dag_module.load_rows(reg, store, "covertype-classifier", "brightness", None, rows=5)
    reg.aliases["covertype-classifier"].pop(Alias.CHAMPION)
    with pytest.raises(AirflowFailException, match="no champion"):
        dag_module.load_rows(reg, store, "covertype-classifier", "shift_feature", None, rows=5)


def test_load_rows_takes_the_first_rows(store):
    reg = registry(store)
    loaded = dag_module.load_rows(reg, store, "covertype-classifier", "shift_feature", None, rows=4)
    assert loaded == {"val_uri": "s3://datasets/covertype-classifier/v1/val.parquet", "rows": 4}


def test_params_and_trigger():
    params = dag_module.DAG.params
    with pytest.raises(ParamValidationError):
        params.get_param("perturbation").resolve("blur")
    with pytest.raises(ParamValidationError):
        params.get_param("duration_minutes").resolve(31)
    trigger = dag_module.DAG.get_task("trigger_drift_check")
    assert trigger.trigger_dag_id == "drift_check"
    assert trigger.trigger_run_id == "simulate__{{ params.model }}__{{ ts_nodash }}"
