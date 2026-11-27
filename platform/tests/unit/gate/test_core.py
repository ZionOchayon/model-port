import dataclasses

import pytest

from modelport.spec import EvaluationSpec, MetricName
from modelport_platform.gate.core import CheckName, EvalReport, GatePolicy, decide

POLICY = GatePolicy.from_spec(EvaluationSpec(min_absolute=0.7))
VAL = "s3://datasets/fashion-mnist/v1/val.parquet"


def report(metric=0.80, per_class=None, latency=10.0, version="2", val_uri=VAL):
    return EvalReport(
        model="fashion-classifier",
        version=version,
        val_uri=val_uri,
        n_samples=600,
        metric_name=MetricName.F1_MACRO,
        metric=metric,
        per_class=per_class or {"Bag": 0.80, "Shirt": 0.80},
        latency_ms_median=latency,
        latency_batch_size=256,
        created_at="2026-11-27T10:00:00Z",
    )


CHAMPION = report(version="1")


def test_first_model_passes_absolute():
    decision = decide(report(0.842), None, POLICY)
    assert decision.passed
    assert [c.name for c in decision.checks] == [CheckName.ABSOLUTE]
    assert decision.checks[0].detail == "f1_macro 0.842 >= min_absolute 0.700"


def test_first_model_fails_absolute():
    decision = decide(report(0.093), None, POLICY)
    assert not decision.passed
    assert decision.reasons == ("f1_macro 0.093 < min_absolute 0.700",)


def test_all_three_pass():
    decision = decide(report(0.81), CHAMPION, POLICY)
    assert decision.passed
    assert [c.name for c in decision.checks] == [CheckName.METRIC, CheckName.PER_CLASS, CheckName.LATENCY]
    assert decision.reasons == ()


def test_metric_below_champion_fails():
    decision = decide(report(0.79), CHAMPION, POLICY)
    assert not decision.passed
    assert decision.reasons == ("f1_macro 0.790 < champion 0.800 + min_improvement 0.000",)


def test_min_improvement_applied():
    policy = dataclasses.replace(POLICY, min_improvement=0.02)
    assert not decide(report(0.81), CHAMPION, policy).passed
    assert decide(report(0.82), CHAMPION, policy).passed


def test_one_class_drop_fails_even_if_mean_improves():
    candidate = report(0.85, per_class={"Bag": 0.98, "Shirt": 0.719})
    decision = decide(candidate, CHAMPION, POLICY)
    assert not decision.passed
    assert decision.reasons == ("class 'Shirt' dropped 0.081 (0.800 -> 0.719) > 0.050",)


def test_missing_class_counts_as_zero():
    decision = decide(report(0.85, per_class={"Bag": 0.9}), CHAMPION, POLICY)
    assert decision.reasons == ("class 'Shirt' dropped 0.800 (0.800 -> 0.000) > 0.050",)


def test_latency_21_percent_fails():
    decision = decide(report(0.81, latency=12.1), CHAMPION, POLICY)
    assert decision.reasons == ("latency 12.1 ms > champion 10.0 ms + 20%",)


def test_latency_20_percent_passes():
    assert decide(report(0.81, latency=12.0), CHAMPION, POLICY).passed


def test_all_failures_reported():
    candidate = report(0.70, per_class={"Bag": 0.70, "Shirt": 0.70}, latency=50.0)
    decision = decide(candidate, CHAMPION, POLICY)
    assert [c.passed for c in decision.checks] == [False, False, False]
    assert len(decision.reasons) == 3
    assert "class 'Bag'" in decision.reasons[1]
    assert "class 'Shirt'" in decision.reasons[1]


def test_mismatched_val_uri_raises():
    with pytest.raises(ValueError, match="val"):
        decide(report(val_uri="s3://datasets/fashion-mnist/v2/val.parquet"), CHAMPION, POLICY)


def test_metric_name_mismatch_raises():
    with pytest.raises(ValueError, match="metric"):
        decide(dataclasses.replace(report(), metric_name=MetricName.ACCURACY), None, POLICY)


def test_report_json_roundtrip():
    r = report()
    payload = r.to_json()
    assert set(payload) == {
        "model",
        "version",
        "val_uri",
        "n_samples",
        "metric_name",
        "metric",
        "per_class",
        "latency_ms_median",
        "latency_batch_size",
        "created_at",
    }
    assert EvalReport.from_json(payload) == r
