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
