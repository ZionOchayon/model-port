import pytest

from modelport.lifecycle import AliasSnapshot
from modelport_platform.metrics import names, push

URL = "http://pushgateway.monitoring:9091"


@pytest.fixture
def pushes(monkeypatch):
    calls = []

    def fake_push(url, *, job, registry, grouping_key):
        series = [(s.name, dict(s.labels), s.value) for m in registry.collect() for s in m.samples]
        samples = {name: value for name, _, value in series}
        calls.append(
            {
                "kind": "put",
                "url": url,
                "job": job,
                "key": dict(grouping_key),
                "samples": samples,
                "series": series,
            }
        )

    def fake_pushadd(*args, **kwargs):
        calls.append({"kind": "pushadd"})

    def fake_delete(url, *, job, grouping_key):
        calls.append({"kind": "delete", "job": job, "key": dict(grouping_key)})

    monkeypatch.setattr(push, "push_to_gateway", fake_push)
    monkeypatch.setattr(push, "pushadd_to_gateway", fake_pushadd, raising=False)
    monkeypatch.setattr(push, "delete_from_gateway", fake_delete)
    return calls


def test_grouping_keys_have_no_run_ids(pushes):
    push.push_gate_result(URL, "m", passed=True)
    push.push_canary_result(URL, "m", passed=False)
    push.push_aliases(URL, "m", AliasSnapshot(model="m", champion="3", previous="2", candidate=None))
    push.push_job_timing(
        URL, model="m", pattern="rayjob", kind="train", startup_s=12.0, duration_s=300.0, cpu_s=None
    )
    keys = [(c["job"], c["key"]) for c in pushes]
    assert keys == [
        ("modelport_lifecycle", {"model": "m", "source": "gate"}),
        ("modelport_lifecycle", {"model": "m", "source": "canary"}),
        ("modelport_lifecycle", {"model": "m", "source": "aliases"}),
        ("modelport_jobs", {"model": "m", "pattern": "rayjob", "kind": "train"}),
    ]


def test_put_semantics_used(pushes):
    push.push_gate_result(URL, "m", passed=False)
    assert [c["kind"] for c in pushes] == ["put"]
    samples = pushes[0]["samples"]
    assert samples[names.GATE_LAST_RESULT] == 0.0
    assert samples[names.GATE_LAST_RESULT_TIMESTAMP] > 1.7e9


def test_alias_versions_and_timings_values(pushes):
    push.push_aliases(URL, "m", AliasSnapshot(model="m", champion="3", previous="2", candidate=None))
    push.push_job_timing(
        URL, model="m", pattern="shared", kind="eval", startup_s=None, duration_s=40.0, cpu_s=None
    )
    aliases, timing = pushes
    assert sorted(aliases["series"], key=lambda s: s[1]["alias"]) == [
        (names.ALIAS_VERSION, {"alias": "champion"}, 3.0),
        (names.ALIAS_VERSION, {"alias": "previous"}, 2.0),
    ]  # no candidate: no series, so a stale candidate disappears with the PUT
    assert set(timing["samples"]) == {names.JOB_DURATION}  # unknown values are not pushed


def test_skipped_without_url(pushes):
    push.push_gate_result(None, "m", passed=True)
    push.delete_model_groups(None, "m")
    assert pushes == []


def test_failure_logged_not_raised(monkeypatch, capsys):
    def broken(*args, **kwargs):
        raise OSError("connection refused")

    monkeypatch.setattr(push, "push_to_gateway", broken)
    push.push_gate_result(URL, "m", passed=True)  # must not raise
    captured = capsys.readouterr()
    assert "push.failed" in captured.out + captured.err


def test_delete_model_groups(pushes):
    push.delete_model_groups(URL, "m")
    assert {(c["job"], tuple(sorted(c["key"].items()))) for c in pushes} >= {
        ("modelport_lifecycle", (("model", "m"), ("source", "gate"))),
        ("modelport_drift", (("model", "m"),)),
    }


def test_push_drift(pushes):
    from modelport.spec import DriftSpec
    from modelport_platform.drift.check import DriftResult

    spec = DriftSpec(psi_warning=0.1, psi_critical=0.25)
    done = DriftResult(
        model="m",
        version="3",
        samples=800,
        psi={"stat_x": 0.3, "prediction": 0.02},
        skipped=False,
        reason=None,
    )
    push.push_drift(URL, "m", done, spec)
    skipped = DriftResult(
        model="m", version="3", samples=0, psi={}, skipped=True, reason="0 rows < min_samples 500"
    )
    push.push_drift(URL, "m", skipped, spec)
    first, second = pushes
    assert (first["job"], first["key"]) == ("modelport_drift", {"model": "m"})
    series = {(name, tuple(sorted(labels.items()))): value for name, labels, value in first["series"]}
    assert series[(names.PSI, (("feature", "stat_x"),))] == 0.3
    assert series[(names.PSI_THRESHOLD, (("level", "critical"),))] == 0.25
    assert series[(names.PSI_THRESHOLD, (("level", "warning"),))] == 0.1
    assert series[(names.DRIFT_SAMPLES, ())] == 800
    assert first["samples"][names.DRIFT_LAST_SUCCESS] > 1.7e9
    # a skipped check replaces the group: samples 0, thresholds kept, no PSI series left behind
    assert names.PSI not in second["samples"]
    assert second["samples"][names.DRIFT_SAMPLES] == 0
    assert names.PSI_THRESHOLD in second["samples"]
