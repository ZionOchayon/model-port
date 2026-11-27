import json

import numpy as np
import pandas as pd

from modelport.spec import parse_spec
from modelport_platform.drift.baseline import Baseline, FeatureKind, build_baseline
from modelport_platform.drift.psi import OTHER_CATEGORY, feature_psi

SPEC = {
    "name": "tiny",
    "class": "m:Tiny",
    "framework": "xgboost",
    "labels": ["low", "high"],
    "signature": {"inputs": [{"name": "x", "type": "double"}, {"name": "k", "type": "long"}]},
    "data": {"dataset": "s3://datasets/tiny"},
}


class Tiny:
    """Stands in for a loaded BaseModel: only run_inference and input_stats are used."""

    def __init__(self, confidence=0.8):
        self.confidence = confidence
        self.spec = parse_spec(SPEC)

    def run_inference(self, batch):
        labels = np.where(batch["x"] > 0.5, "high", "low")
        return pd.DataFrame({"prediction": labels, "confidence": np.full(len(batch), self.confidence)})

    def input_stats(self, batch):
        return batch[["x", "k"]].astype(np.float64)


def frame(n=1000, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"x": rng.random(n), "k": rng.integers(0, 3, n)})


def baseline(val=None, **kw):
    return build_baseline(
        Tiny(**kw), frame() if val is None else val, model_name="tiny", version="3", bins=10
    )


def test_continuous_edges_open_ended():
    feature = baseline().features["stat_x"]
    assert feature.kind is FeatureKind.CONTINUOUS
    assert feature.edges[0] == -np.inf
    assert feature.edges[-1] == np.inf
    assert len(feature.shares) == len(feature.edges) - 1 == 10
    assert abs(sum(feature.shares) - 1.0) < 1e-9


def test_duplicate_edges_collapsed():
    feature = baseline().features["stat_k"]  # three distinct values -> fewer than 10 bins
    assert len(feature.edges) == len(set(feature.edges))
    assert len(feature.shares) < 10
    assert feature_psi(feature, frame(seed=1)["k"].to_numpy(dtype=float)) < 0.1


def test_categorical_includes_other():
    feature = baseline().features["prediction"]
    assert feature.kind is FeatureKind.CATEGORICAL
    assert feature.categories == ("low", "high")
    assert len(feature.shares) == 3  # + __other__
    assert feature_psi(feature, [OTHER_CATEGORY] * 100) > 0.25


def test_json_roundtrip():
    b = baseline()
    payload = json.loads(json.dumps(b.to_json(), allow_nan=False))  # JSON has no infinity
    assert payload["features"]["stat_x"]["edges"][0] == -1e308
    assert Baseline.from_json(payload) == b
    assert set(payload) == {"model", "version", "n_rows", "confidence_mean", "class_shares", "features"}


def test_confidence_mean():
    b = baseline(confidence=0.7)
    assert abs(b.confidence_mean - 0.7) < 1e-12
    assert b.features["confidence"].mean == b.confidence_mean
    assert set(b.class_shares) == {"low", "high"}
    assert abs(sum(b.class_shares.values()) - 1.0) < 1e-9
    assert (b.model, b.version, b.n_rows) == ("tiny", "3", 1000)


def test_rows_capped():
    assert baseline(frame(6000)).n_rows == 5000
