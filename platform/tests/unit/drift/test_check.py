import io
from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from modelport_platform.drift.baseline import Baseline, FeatureBaseline, FeatureKind
from modelport_platform.drift.check import compute_drift, load_inference_logs
from modelport_platform.drift.psi import categorical_shares, quantile_edges, shares

LABELS = ("cat", "dog")


def continuous(name, values):
    edges = quantile_edges(values, 10)
    return FeatureBaseline(
        name=name,
        kind=FeatureKind.CONTINUOUS,
        edges=tuple(edges),
        categories=None,
        shares=tuple(shares(values, edges)),
        mean=float(np.mean(values)),
    )


def reference(n=5000, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "stat_brightness": rng.normal(120, 20, n),
            "stat_contrast": rng.normal(40, 5, n),
            "confidence": rng.uniform(0.5, 1.0, n),
            "prediction": rng.choice(LABELS, n),
        }
    )


def baseline(ref):
    features = {
        c: continuous(c, ref[c].to_numpy()) for c in ("stat_brightness", "stat_contrast", "confidence")
    }
    features["prediction"] = FeatureBaseline(
        name="prediction",
        kind=FeatureKind.CATEGORICAL,
        edges=None,
        categories=LABELS,
        shares=tuple(categorical_shares(ref["prediction"].tolist(), LABELS)),
        mean=None,
    )
    return Baseline(
        model="m", version="3", n_rows=len(ref), features=features, confidence_mean=0.75, class_shares={}
    )


def logs(ref, version="3"):
    return ref.assign(version=version, model="m", variant="champion")


def test_skip_below_min_samples():
    ref = reference()
    result = compute_drift(logs(ref.head(412)), baseline(ref), version="3", min_samples=500)
    assert result.skipped
    assert result.samples == 412
    assert result.psi == {}
    assert result.reason == "412 rows < min_samples 500"


def test_no_drift_low_psi():
    ref = reference()
    result = compute_drift(logs(reference(2000, seed=1)), baseline(ref), version="3", min_samples=500)
    assert not result.skipped
    assert result.samples == 2000
    assert set(result.psi) == {"stat_brightness", "stat_contrast", "confidence", "prediction"}
    assert max(result.psi.values()) < 0.1


def test_shift_detected_on_right_feature():
    ref = reference()
    shifted = reference(2000, seed=1)
    shifted["stat_brightness"] = shifted["stat_brightness"] * 2.0
    result = compute_drift(logs(shifted), baseline(ref), version="3", min_samples=500)
    assert max(result.psi, key=result.psi.get) == "stat_brightness"
    assert result.psi["stat_brightness"] > 0.25
    assert result.psi["stat_contrast"] < 0.1


def test_filters_other_versions():
    ref = reference()
    rows = pd.concat([logs(reference(600, seed=1)), logs(reference(900, seed=2), version="2")])
    result = compute_drift(rows, baseline(ref), version="3", min_samples=500)
    assert result.samples == 600


def test_load_inference_logs_window(store):
    end = datetime(2026, 1, 2, 6, tzinfo=UTC)

    def write(day, name, times):
        table = pa.table(
            {
                "ts": pa.array(times, pa.timestamp("ms", tz="UTC")),
                "version": ["3"] * len(times),
                "confidence": [0.9] * len(times),
            }
        )
        buffer = io.BytesIO()
        pq.write_table(table, buffer)
        store.write_bytes(f"s3://inference-logs/tiny-model/{day}/{name}", buffer.getvalue())

    write("2026-01-01", "champion-v3-a-1.parquet", [end - timedelta(hours=30), end - timedelta(hours=20)])
    write("2026-01-02", "champion-v3-a-2.parquet", [end - timedelta(hours=1), end + timedelta(hours=1)])
    write("2026-01-02", "notes.txt", [end])  # not a parquet file: ignored
    frame = load_inference_logs(store, "tiny-model", end - timedelta(hours=24), end)
    assert sorted(frame["ts"]) == [end - timedelta(hours=20), end - timedelta(hours=1)]
    assert load_inference_logs(store, "other-model", end - timedelta(hours=24), end).empty
