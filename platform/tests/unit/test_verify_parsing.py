import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
import verify


def test_parse_checks():
    assert verify.parse_checks("V-01, V-06") == ["V-01", "V-06"]
    with pytest.raises(SystemExit):
        verify.parse_checks("V-99")


def test_default_stage_from_changelog():
    assert verify.default_stage("# Changelog\n\n## [Unreleased]\n\n### Added\n") == "v0.1"
    assert verify.default_stage("## [Unreleased]\n\n## [0.1.0] - 2026-12-01\n") == "v0.1"
    assert verify.default_stage("## [Unreleased]\n\n## [0.2.0] - 2026-12-09\n\n## [0.1.0] - x\n") == "v0.2"


def test_checks_for_stage():
    assert "V-09" not in verify.checks_for_stage("v0.1")
    assert {"V-01", "V-07b", "V-08"} <= set(verify.checks_for_stage("v0.1"))
    assert "V-12" in verify.checks_for_stage("v0.2")


def test_zip_matches_folder():
    folder = {"model.py", "model.yaml", "example/input.parquet"}
    entries = ["model/model.py", "model/model.yaml", "model/example/input.parquet"]
    assert verify.zip_extras(entries, folder) == []
    assert verify.zip_extras([*entries, "model/secret.env", "other/x.py"], folder) == [
        "model/secret.env",
        "other/x.py",
    ]


def test_steps_complete():
    assert verify.missing_steps([0, 1], epochs=2) == []
    assert verify.missing_steps([1, 1], epochs=3) == [0, 2]


def test_missing_lineage():
    tags = {"modelport.dataset": "fashion-mnist", "modelport.image": ""}
    missing = verify.missing_lineage(tags)
    assert "modelport.image" in missing
    assert "modelport.code_sha256" in missing
    assert "modelport.dataset" not in missing


def test_report_missing_keys():
    report = {"model": "m", "version": "1", "metric": 0.9}
    assert "latency_ms_median" in verify.report_missing_keys(report)
    assert verify.report_missing_keys({k: 1 for k in verify.REPORT_KEYS}) == []


def test_result_line_format(capsys):
    assert verify.report_results([("V-01", None), ("V-06", "tag modelport.image empty")]) == 1
    out = capsys.readouterr().out
    assert "PASS V-01" in out
    assert "FAIL V-06: tag modelport.image empty" in out


def test_ops_run_state_and_row_hashes():
    import ops
    import pandas as pd

    assert ops.run_state_done("success", "success") is True
    assert ops.run_state_done("running", "success") is False
    with pytest.raises(SystemExit):
        ops.run_state_done("failed", "success")
    a = pd.DataFrame(
        {"x": [1, 2], "prediction": ["p", "q"], "confidence": [0.5, 0.25], "model": "m", "version": "1"}
    )
    b = a.iloc[::-1].reset_index(drop=True)  # same rows, other order
    assert ops.row_hashes(a) == ops.row_hashes(b)
    c = a.assign(confidence=[0.5, 0.26])
    assert ops.row_hashes(a) != ops.row_hashes(c)


def test_snapshot_has_checkpoint():
    import ops

    assert ops.snapshot_has_checkpoint(
        '{"checkpoint_results": [{"checkpoint_dir_name": "checkpoint_2026-11-27_10"}]}'
    )
    assert not ops.snapshot_has_checkpoint('{"checkpoint_results": []}')
    assert not ops.snapshot_has_checkpoint("not json")


def test_resume_evidence():
    assert verify.resume_problems([0, 1, 2, 3], epochs=4, resumed_from=[0, 0, 2, 2]) == []
    assert "logged 2 times" in verify.resume_problems([0, 0, 1, 2, 3], epochs=4, resumed_from=[0, 0, 0, 0])[0]
    assert verify.resume_problems([0, 1, 2, 3], epochs=4, resumed_from=[0, 0, 0, 0]) == [
        "no epoch resumed from a checkpoint (resumed_from stayed 0)"
    ]
    assert verify.resume_problems([0, 1], epochs=4, resumed_from=[0, 1]) == ["epoch steps [2, 3] missing"]


def test_pip_by_app():
    config = """
applications:
  - {name: fashion-classifier, route_prefix: /fashion-classifier}
  - name: fashion-classifier--champion
    runtime_env: {env_vars: {A: b}}
  - name: covertype-classifier--champion
    runtime_env: {pip: {packages: [xgboost==3.4.1], pip_check: false}}
"""
    assert verify.pip_by_app(config) == {
        "fashion-classifier": (),
        "fashion-classifier--champion": (),
        "covertype-classifier--champion": ("xgboost==3.4.1",),
    }


def test_zero_downtime_problems():
    good = {
        "sent": 1500,
        "failed": 0,
        "champion_before": "4",
        "champion_after": "5",
        "promoted_during_load": True,
    }
    assert verify.zero_downtime_problems(good) == []
    bad = {**good, "sent": 120, "failed": 2, "status_counts": {"200": 118, "conn": 2}, "champion_after": "4"}
    assert verify.zero_downtime_problems(bad) == [
        "2 of 120 requests failed ({'200': 118, 'conn': 2})",
        "only 120 requests sent (at least 300)",
        "the champion stayed v4",
    ]
    late = {**good, "promoted_during_load": False}
    assert verify.zero_downtime_problems(late) == ["the promotion finished after the load ended"]


def test_canary_failure_problems():
    tags = {
        "modelport.canary.result": "failed",
        "modelport.canary.reasons": '["error rate 0.480 > max_error_rate 0.010"]',
    }
    aliases = {"champion": "1", "previous": None, "candidate": None}
    ok = {
        "state": "rejected",
        "tags": tags,
        "aliases": aliases,
        "champion_before": "1",
        "apps": ["m", "m--champion"],
    }
    assert verify.canary_failure_problems("m", "2", **ok) == []
    bad = {
        **ok,
        "state": "candidate",
        "tags": {
            "modelport.canary.result": "failed",
            "modelport.canary.reasons": '["only 3 requests (need 50)"]',
        },
        "aliases": {**aliases, "candidate": "2", "champion": "2"},
        "apps": ["m", "m--champion", "m--candidate"],
    }
    assert verify.canary_failure_problems("m", "2", **bad) == [
        "v2 is candidate, not rejected",
        "no error-rate reason: ['only 3 requests (need 50)']",
        "v2 still has the alias candidate, champion",
        "the champion changed from v1 to v2",
        "the RayService still serves m--candidate",
    ]


def test_ops_wait_dag_run_if_exists(monkeypatch):
    import ops

    class Absent:
        def get_run(self, dag_id, run_id):
            return None

    monkeypatch.setattr(ops, "airflow", Absent)
    wait = ["wait-dag-run", "deploy_candidate", "deploy__m__v3"]
    assert ops.main([*wait, "--if-exists", "--timeout", "1"]) == 0
    assert ops.main([*wait, "--timeout", "0"]) != 0


def test_batch_rerun_problems():
    run = {"m": {"rows": 2, "hashes": ["a", "b"]}, "n": {"rows": 1, "hashes": ["c"]}}
    inputs = {"m": 2, "n": 1}
    assert verify.batch_rerun_problems(run, json.loads(json.dumps(run)), inputs) == []
    other = {**run, "n": {"rows": 1, "hashes": ["d"]}}
    assert verify.batch_rerun_problems(run, other, inputs) == ["n: the two runs wrote different predictions"]
    assert verify.batch_rerun_problems(run, run, {"m": 3, "n": 1}) == ["m: 2 predictions for 3 input rows"]
    assert verify.batch_rerun_problems({}, {}, inputs) == ["m: no predictions", "n: no predictions"]
