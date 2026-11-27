import io
import json
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

pytest.importorskip("airflow.sdk")
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "dags"))
import ingest_dataset as dag_module

PNG = b"\x89PNG\r\n\x1a\n" + b"x"
CONTRACT = {
    "dataset": "fashion-mnist",
    "label_column": "label",
    "columns": {"image": "binary", "label": "string"},
    "min_rows": 4,
    "image_format": "png",
}


class Events:
    def __init__(self):
        self.added = []

    def __getitem__(self, alias):
        events = self

        class Accessor:
            def add(self, asset, extra):
                events.added.append((alias.name, asset.uri, extra))

        return Accessor()


def parquet(rows=8):
    labels = ["Bag", "Coat"] * (rows // 2)
    buffer = io.BytesIO()
    pq.write_table(pa.table({"image": pa.array([PNG] * rows, pa.binary()), "label": labels}), buffer)
    return buffer.getvalue()


@pytest.fixture
def ready(store):
    store.write_bytes("s3://datasets/fashion-mnist/contract.json", json.dumps(CONTRACT).encode())
    return store


def item(uri):
    return {"dataset": uri.split("/")[3], "uri": uri}


def test_process_file_valid_creates_version_and_event(ready):
    ready.write_bytes("s3://incoming/fashion-mnist/a.parquet", parquet())
    events = Events()
    assert dag_module.list_incoming(ready) == [item("s3://incoming/fashion-mnist/a.parquet")]
    result = dag_module.process_file(item("s3://incoming/fashion-mnist/a.parquet"), ready, events)
    assert (result["version"], result["created"]) == (1, True)
    assert ready.exists("s3://datasets/fashion-mnist/v1/manifest.json")
    assert not ready.exists("s3://incoming/fashion-mnist/a.parquet")
    ((alias, uri, extra),) = events.added
    assert (alias, uri) == ("modelport-datasets", "s3://datasets/fashion-mnist")
    assert extra["version"] == 1


def test_process_file_corrupt_quarantines(ready):
    data = parquet()[:50]
    ready.write_bytes("s3://incoming/fashion-mnist/bad.parquet", data)
    events = Events()
    result = dag_module.process_file(item("s3://incoming/fashion-mnist/bad.parquet"), ready, events)
    assert result["quarantined"].startswith("s3://quarantine/fashion-mnist/")
    report = json.loads(ready.read_bytes(result["quarantined"] + "report.json"))
    assert report["checks"][0] == {
        "name": "readable",
        "passed": False,
        "detail": report["checks"][0]["detail"],
    }
    assert not ready.exists("s3://incoming/fashion-mnist/bad.parquet")
    assert events.added == []


def test_process_file_duplicate_no_event(ready):
    for name in ("a", "b"):
        ready.write_bytes(f"s3://incoming/fashion-mnist/{name}.parquet", parquet())
    events = Events()
    dag_module.process_file(item("s3://incoming/fashion-mnist/a.parquet"), ready, events)
    again = dag_module.process_file(item("s3://incoming/fashion-mnist/b.parquet"), ready, events)
    assert (again["version"], again["created"]) == (1, False)
    assert len(events.added) == 1
    assert not ready.exists("s3://incoming/fashion-mnist/b.parquet")


def test_no_contract_quarantines(store):
    store.write_bytes("s3://incoming/shapes/a.parquet", parquet())
    result = dag_module.process_file(item("s3://incoming/shapes/a.parquet"), store, Events())
    report = json.loads(store.read_bytes(result["quarantined"] + "report.json"))
    assert [c["name"] for c in report["checks"] if not c["passed"]] == ["schema"]
    assert "no contract for dataset 'shapes'" in report["checks"][0]["detail"]


def test_invalid_dataset_name_quarantined_under_unknown(store):
    store.write_bytes("s3://incoming/Bad_Name/a.parquet", parquet())
    result = dag_module.process_file(item("s3://incoming/Bad_Name/a.parquet"), store, Events())
    assert result["quarantined"].startswith("s3://quarantine/unknown-dataset/")


def test_continuous_runs_get_distinct_logical_dates():
    # without a start_date the first run's interval is empty and every later run reuses its logical date
    from airflow.timetables.base import TimeRestriction
    from airflow.timetables.simple import ContinuousTimetable

    dag = dag_module.ingest_dataset()
    restriction = TimeRestriction(earliest=dag.start_date, latest=None, catchup=False)
    first = ContinuousTimetable().next_dagrun_info(last_automated_data_interval=None, restriction=restriction)
    second = ContinuousTimetable().next_dagrun_info(
        last_automated_data_interval=first.data_interval, restriction=restriction
    )
    assert second.logical_date > first.logical_date
