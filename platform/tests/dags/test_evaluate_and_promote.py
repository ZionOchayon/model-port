import importlib.util
import json
import sys
from pathlib import Path

import pytest

from modelport.lifecycle import ModelState
from modelport.naming import Alias, Tag

pytest.importorskip("airflow.sdk")
from airflow.sdk.exceptions import AirflowSkipException

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "dags"))
import evaluate_and_promote as dag_module

ROOT = Path(__file__).resolve().parents[3]
SPEC = (ROOT / "examples" / "models" / "image-classifier" / "model.fashion.yaml").read_bytes()
_spec = importlib.util.spec_from_file_location("sdk_fakes", ROOT / "sdk/tests/fakes.py")
sdk_fakes = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sdk_fakes)
MODEL = "fashion-classifier"
V1_VAL = "s3://datasets/fashion-mnist/v1/val.parquet"
V2_VAL = "s3://datasets/fashion-mnist/v2/val.parquet"


def registry(champion=True):
    reg = sdk_fakes.InMemoryRegistry()
    reg.ensure_registered_model(MODEL, {})
    for n, val in ((1, V1_VAL), (2, V2_VAL)):
        run = f"train-{n}"
        reg.runs[run] = {"tags": {}, "artifacts": {"spec/model.yaml": SPEC}}
        tags = {Tag.VAL_URI: val, Tag.TRAIN_RUN_ID: run, Tag.RUNTIME_PIP: '["torch==2.15.0"]'}
        reg.add_version(MODEL, {**tags, Tag.STATE: "champion" if n == 1 and champion else "registered"})
    if champion:
        reg.aliases[MODEL][Alias.CHAMPION] = "1"
    return reg


def test_prepare_uses_champion_val_set(tmp_path):
    reg = registry()
    out = dag_module.prepare({"model": MODEL, "version": "2"}, "manual__x", reg, tmp_path)
    assert (out["version"], out["champion"], out["val_uri"]) == ("2", "1", V1_VAL)
    assert out["candidate_run"] != out["champion_run"]
    assert reg.runs[out["candidate_run"]]["tags"][Tag.EVAL_KEY] == f"{MODEL}/2/{V1_VAL}/manual__x"
    assert out["runtime_pip"] == {"2": ["torch==2.15.0"], "1": ["torch==2.15.0"]}
    assert out["auto_deploy"] is True
    first = dag_module.prepare({"model": MODEL, "version": "2"}, "manual__x", reg, tmp_path)
    assert first["candidate_run"] == out["candidate_run"]  # a retry reuses the eval runs
    alone = dag_module.prepare({"model": MODEL, "version": "2"}, "r", registry(champion=False), tmp_path)
    assert (alone["champion"], alone["val_uri"], alone["champion_run"]) == (None, V2_VAL, None)


def test_rerun_on_decided_version_skips(tmp_path):
    reg = registry()
    reg.set_version_tags(MODEL, "2", {Tag.STATE: ModelState.REJECTED})
    with pytest.raises(AirflowSkipException, match="v2 is already rejected"):
        dag_module.prepare({"model": MODEL, "version": "2"}, "r", reg, tmp_path)


class Jobs:
    def __init__(self):
        self.submitted = []

    def submit_idempotent(self, **kwargs):
        self.submitted.append(kwargs)
        return kwargs["submission_id"]

    def wait(self, submission_id, *, timeout_s, poll_s=5.0, submitted_at=None):
        from modelport_platform.ray.jobs import JobTimings

        return JobTimings(submitted_at, None, None)


def test_evaluation_job_submission(platform_settings):
    jobs = Jobs()
    dag_module.run_evaluation(
        jobs,
        platform_settings,
        model=MODEL,
        version="2",
        val_uri=V1_VAL,
        report_run="run-9",
        runtime_pip=["torch==2.15.0"],
        cpus=1.0,
        dag_run_id="manual__x",
    )
    (job,) = jobs.submitted
    assert job["submission_id"].startswith("eval__fashion-classifier__v2-")
    assert job["entrypoint"] == (
        "python -m modelport_platform.gate.evaluate --model fashion-classifier --version 2"
        f" --val-uri {V1_VAL} --report-run-id run-9"
    )
    assert job["runtime_env"]["pip"] == {"packages": ["torch==2.15.0"], "pip_check": False}
    assert job["runtime_env"]["env_vars"]["MLFLOW_TRACKING_URI"] == platform_settings.mlflow_uri
    assert "AWS_SECRET_ACCESS_KEY" not in job["runtime_env"]["env_vars"]  # secrets come from the pod
    assert job["entrypoint_num_cpus"] == 1.0


def report(version, metric, per_class=None):
    return {
        "model": MODEL,
        "version": version,
        "val_uri": V1_VAL,
        "n_samples": 600,
        "metric_name": "f1_macro",
        "metric": metric,
        "per_class": per_class or {"Bag": metric},
        "latency_ms_median": 10.0,
        "latency_batch_size": 256,
        "created_at": "2026-11-27T10:00:00Z",
    }


def decided(reg, candidate_metric, champion=True, auto_deploy=True):
    reg.log_json("eval-c", "gate/report.json", report("2", candidate_metric))
    if champion:
        reg.log_json("eval-p", "gate/report.json", report("1", 0.80))
    prepared = {
        "version": "2",
        "champion": "1" if champion else None,
        "val_uri": V1_VAL,
        "candidate_run": "eval-c",
        "champion_run": "eval-p" if champion else None,
        "auto_deploy": auto_deploy,
    }
    return dag_module.decide_version(reg, MODEL, prepared, pushgateway_url=None)


def test_decide_pass_sets_candidate_and_tags():
    reg = registry()
    out = decided(reg, 0.85)
    assert out == {"passed": True, "reasons": [], "auto_deploy": True}
    info = reg.get_version(MODEL, "2")
    assert info.tags[Tag.GATE_RESULT] == "passed"
    assert json.loads(info.tags[Tag.GATE_REASONS]) == []
    assert info.tags[Tag.GATE_EVAL_RUN_ID] == "eval-c"
    assert reg.snapshot(MODEL).candidate == "2"
    assert info.state is ModelState.CANDIDATE


def test_decide_fail_sets_rejected_with_reasons():
    reg = registry()
    out = decided(reg, 0.70)
    assert out["passed"] is False
    assert out["reasons"][0].startswith("f1_macro 0.700 < champion 0.800")
    info = reg.get_version(MODEL, "2")
    assert info.tags[Tag.GATE_RESULT] == "failed"
    assert json.loads(info.tags[Tag.GATE_REASONS]) == out["reasons"]
    assert info.state is ModelState.REJECTED
    assert reg.snapshot(MODEL).candidate is None


def test_first_model_absolute_bar():
    assert decided(registry(champion=False), 0.85, champion=False)["passed"] is True


def test_no_auto_deploy_short_circuits():
    assert dag_module.should_deploy({"passed": True, "auto_deploy": False}) is False
    assert dag_module.should_deploy({"passed": False, "auto_deploy": True}) is False
    expected = (Path(dag_module.__file__).with_name("deploy_candidate.py")).exists()
    assert dag_module.should_deploy({"passed": True, "auto_deploy": True}) is expected


def test_structure():
    tasks = {t.task_id: t for t in dag_module.DAG.tasks}
    assert set(tasks) == {
        "prepare",
        "evaluate_candidate",
        "evaluate_champion",
        "decide",
        "should_deploy",
        "trigger_deploy",
    }
    assert tasks["evaluate_candidate"].pool == tasks["evaluate_champion"].pool == "shared_cluster"
    assert dag_module.DAG.max_active_runs == 2


def test_decide_skips_when_prepare_skipped():
    tasks = {t.task_id: t for t in dag_module.DAG.tasks}
    assert tasks["decide"].trigger_rule == "none_failed_min_one_success"
