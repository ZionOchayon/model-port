import io
import json

import pyarrow as pa
import pyarrow.parquet as pq

from modelport_platform.ingest.quality import DatasetContract, check_table
from modelport_platform.ingest.versioning import (
    find_version_by_sha,
    latest_version,
    load_contract,
    quarantine,
    split_train_val,
    write_version,
)

DS = "fashion-mnist"
SHA_A, SHA_B = "a" * 64, "b" * 64
CONTRACT = DatasetContract.from_json(
    {"dataset": DS, "label_column": "label", "columns": {"x": "double", "label": "string"}, "min_rows": 10}
)


def table(rows=100):
    return pa.table(
        {"x": [float(i) for i in range(rows)], "label": ["a" if i % 2 else "b" for i in range(rows)]}
    )


def report(t, sha=SHA_A):
    return check_table(t, None, CONTRACT, source_uri=f"s3://incoming/{DS}/f.parquet", sha256=sha)


def write(store, t, sha=SHA_A):
    return write_version(store, DS, t, sha, f"s3://incoming/{DS}/f.parquet", report(t, sha))


def test_first_version_is_v1(store):
    v = write(store, table())
    assert (v.version, v.uri, v.created) == (1, "s3://datasets/fashion-mnist/v1", True)
    assert (v.rows_train, v.rows_val, v.labels) == (90, 10, ("a", "b"))
    manifest = json.loads(store.read_bytes("s3://datasets/fashion-mnist/v1/manifest.json"))
    assert manifest["version"] == 1
    assert manifest["sha256"] == SHA_A
    assert manifest["class_counts"] == {"a": 50, "b": 50}
    assert manifest["quality"]["passed"] is True
    assert manifest["created_at"].endswith("Z")


def test_same_sha_returns_existing_version(store):
    first = write(store, table())
    again = write(store, table())
    assert (again.version, again.created) == (first.version, False)
    assert find_version_by_sha(store, DS, SHA_A).version == 1
    assert find_version_by_sha(store, DS, SHA_B) is None


def test_next_version_increments(store):
    write(store, table())
    v2 = write(store, table(120), sha=SHA_B)
    assert v2.version == 2
    assert latest_version(store, DS).version == 2


def test_split_deterministic():
    t = table()
    _, a_val = split_train_val(t, SHA_A)
    _, b_val = split_train_val(t, SHA_A)
    assert a_val.column("x").to_pylist() == b_val.column("x").to_pylist()
    _, other_val = split_train_val(t, SHA_B)
    assert a_val.column("x").to_pylist() != other_val.column("x").to_pylist()


def test_split_sizes():
    train, val = split_train_val(table(101), SHA_A)
    assert (train.num_rows, val.num_rows) == (90, 11)  # ceil(101 * 0.1)
    assert sorted(train.column("x").to_pylist() + val.column("x").to_pylist()) == [
        float(i) for i in range(101)
    ]


def test_manifest_written_last(store):
    write(store, table())
    assert store.writes[-1] == "s3://datasets/fashion-mnist/v1/manifest.json"
    assert set(store.writes[:-1]) == {
        "s3://datasets/fashion-mnist/v1/train.parquet",
        "s3://datasets/fashion-mnist/v1/val.parquet",
    }
    assert (
        pq.read_table(io.BytesIO(store.read_bytes("s3://datasets/fashion-mnist/v1/train.parquet"))).num_rows
        == 90
    )


def test_folder_without_manifest_overwritten(store):
    store.write_bytes("s3://datasets/fashion-mnist/v1/train.parquet", b"half-written")
    assert latest_version(store, DS) is None
    v = write(store, table())
    assert v.version == 1
    assert store.read_bytes("s3://datasets/fashion-mnist/v1/train.parquet") != b"half-written"


def test_quarantine_writes_report(store):
    store.write_bytes(f"s3://incoming/{DS}/bad.parquet", b"not parquet")
    bad = check_table(
        None, "not a parquet file", CONTRACT, source_uri=f"s3://incoming/{DS}/bad.parquet", sha256=SHA_B
    )
    target = quarantine(store, DS, f"s3://incoming/{DS}/bad.parquet", SHA_B, bad)
    assert target == f"s3://quarantine/{DS}/{SHA_B}/"
    assert store.read_bytes(f"s3://quarantine/{DS}/{SHA_B}/bad.parquet") == b"not parquet"
    report_json = json.loads(store.read_bytes(f"s3://quarantine/{DS}/{SHA_B}/report.json"))
    assert report_json["passed"] is False


def test_load_contract(store):
    store.write_bytes(
        "s3://datasets/fashion-mnist/contract.json",
        json.dumps({"dataset": DS, "label_column": "label", "columns": {"label": "string"}}).encode(),
    )
    assert load_contract(store, DS).label_column == "label"
