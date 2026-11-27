import io
from datetime import UTC, datetime

import pandas as pd
import pyarrow.parquet as pq

from modelport_platform.serve.inference_log import InferenceLogBuffer

T0 = datetime(2026, 11, 27, 9, 0, tzinfo=UTC).timestamp()


class Clock:
    def __init__(self):
        self.now = T0

    def __call__(self):
        return self.now


def rows(n):
    stats = pd.DataFrame({"brightness": [0.5] * n, "contrast": [0.1] * n})
    output = pd.DataFrame({"prediction": ["Bag"] * n, "confidence": [0.9] * n})
    return stats, output


def buffer(store, clock, **kwargs):
    return InferenceLogBuffer(
        model="fashion-classifier",
        version="3",
        variant="champion",
        store=store,
        replica="r1",
        clock=clock,
        **kwargs,
    )


def test_flush_on_rows(store):
    log = buffer(store, Clock(), flush_rows=4)
    log.add(*rows(3), request_id="a")
    assert store.objects == {}
    log.add(*rows(2), request_id="b")
    (uri,) = store.objects
    assert uri == "s3://inference-logs/fashion-classifier/2026-11-27/champion-v3-r1-1795770000000.parquet"
    assert pq.read_table(io.BytesIO(store.objects[uri])).num_rows == 5


def test_flush_on_time(store):
    clock = Clock()
    log = buffer(store, clock, flush_rows=500, flush_seconds=60)
    log.add(*rows(1), request_id="a")
    clock.now += 61
    log.add(*rows(1), request_id="b")
    assert len(store.objects) == 1
    assert log.flush() is None  # nothing left


def test_sample_rate_zero_writes_nothing(store):
    log = buffer(store, Clock(), flush_rows=1, sample_rate=0.0)
    log.add(*rows(10), request_id="a")
    assert log.flush() is None
    assert store.objects == {}


def test_flush_error_does_not_raise(store):
    def broken(uri, data, *, content_type="application/octet-stream"):
        raise ConnectionError("s3 down")

    store.write_bytes = broken
    log = buffer(store, Clock(), flush_rows=1)
    log.add(*rows(2), request_id="a")  # flush fails inside add: logged, rows dropped
    assert log.flush() is None


def test_columns_match_contract(store):
    log = buffer(store, Clock())
    log.add(*rows(2), request_id="req-1")
    uri = log.flush()
    table = pq.read_table(io.BytesIO(store.objects[uri]))
    assert table.column_names == [
        "ts",
        "request_id",
        "model",
        "version",
        "variant",
        "prediction",
        "confidence",
        "stat_brightness",
        "stat_contrast",
    ]
    assert str(table.schema.field("ts").type) == "timestamp[ms, tz=UTC]"
    frame = table.to_pandas()
    assert frame["request_id"].tolist() == ["req-1", "req-1"]
    assert frame["confidence"].dtype == "float64"
