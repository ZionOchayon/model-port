import importlib.util
import io
import json
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from modelport.naming import Alias, Tag

pytest.importorskip("airflow.sdk")
from airflow.sdk.exceptions import ParamValidationError

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "dags"))
import drift_check as dag_module

ROOT = Path(__file__).resolve().parents[3]
SPEC = (ROOT / "examples/models/covertype-classifier/model.yaml").read_bytes()
_spec = importlib.util.spec_from_file_location("sdk_fakes", ROOT / "sdk/tests/fakes.py")
sdk_fakes = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sdk_fakes)
END = datetime(2026, 1, 2, 6, tzinfo=UTC)
BASELINE = {
    "model": "covertype-classifier",
    "version": "1",
    "n_rows": 1000,
    "confidence_mean": 0.8,
    "class_shares": {},
    "features": {
        "confidence": {"kind": "continuous", "edges": [-1e308, 0.5, 1e308], "shares": [0.5, 0.5], "mean": 0.5}
    },
}


def registry():
    reg = sdk_fakes.InMemoryRegistry()
    for model in ("covertype-classifier", "unserved"):
        reg.ensure_registered_model(model, {})
        run = f"{model}-train-1"
        reg.runs[run] = {
            "tags": {},
            "artifacts": {"spec/model.yaml": SPEC, "drift/baseline.json": json.dumps(BASELINE).encode()},
        }
        reg.add_version(model, {Tag.TRAIN_RUN_ID: run})
    reg.aliases["covertype-classifier"][Alias.CHAMPION] = "1"
    return reg


def test_plan_only_champions():
    reg = registry()
    assert dag_module.plan(reg, None) == ["covertype-classifier"]
    assert dag_module.plan(reg, ["unserved"]) == []


def test_skipped_push_has_no_psi(store, monkeypatch):
    pushed = []
    monkeypatch.setattr(dag_module, "push_drift", lambda url, model, result, spec: pushed.append(result))
    summary = dag_module.check_model(
        registry(), store, "http://pg:9091", "covertype-classifier", END, hours=24
    )
    assert summary == {
        "model": "covertype-classifier",
        "version": "1",
        "samples": 0,
        "skipped": True,
        "reason": "0 rows < min_samples 500",
        "max_psi": None,
    }
    (result,) = pushed
    assert result.skipped
    assert result.psi == {}


def test_check_model_computes_psi(store, monkeypatch):
    pushed = []
    monkeypatch.setattr(dag_module, "push_drift", lambda url, model, result, spec: pushed.append(result))
    times = [END - timedelta(minutes=i) for i in range(600)]
    table = pa.table(
        {
            "ts": pa.array(times, pa.timestamp("ms", tz="UTC")),
            "version": ["1"] * 600,
            "confidence": [0.9] * 600,  # every row in the upper bin: drift
        }
    )
    buffer = io.BytesIO()
    pq.write_table(table, buffer)
    store.write_bytes(
        "s3://inference-logs/covertype-classifier/2026-01-02/champion-v1-a-1.parquet", buffer.getvalue()
    )
    summary = dag_module.check_model(registry(), store, None, "covertype-classifier", END, hours=24)
    assert summary["samples"] == 600
    assert summary["skipped"] is False
    assert summary["max_psi"] > 1.0
    assert pushed[0].psi.keys() == {"confidence"}


def test_window_param_bounds():
    params = dag_module.DAG.params
    for bad in (0, 169):
        with pytest.raises(ParamValidationError):
            params.get_param("window_hours").resolve(bad)
    assert params.get_param("window_hours").resolve(24) == 24
    assert dag_module.window_end(None, END) == END
    assert dag_module.window_end("2026-01-01T12:00:00+00:00", END) == datetime(2026, 1, 1, 12, tzinfo=UTC)
