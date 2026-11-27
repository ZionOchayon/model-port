import importlib.util
import io
import itertools
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import yaml

from modelport.lifecycle import ModelState
from modelport.naming import Alias, Tag

pytest.importorskip("airflow.sdk")
from airflow.sdk.exceptions import AirflowSkipException

from modelport_platform import dagkit
from modelport_platform.settings import PlatformSettings

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "dags"))
import deploy_candidate as dag_module

ROOT = Path(__file__).resolve().parents[3]
SPEC = (ROOT / "examples" / "models" / "covertype-classifier" / "model.yaml").read_bytes()
_spec = importlib.util.spec_from_file_location("sdk_fakes", ROOT / "sdk/tests/fakes.py")
sdk_fakes = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sdk_fakes)
MODEL = "covertype-classifier"
NOW = datetime(2026, 11, 27, 12, 0, tzinfo=UTC)
SETTINGS = PlatformSettings.from_env(
    {
        "MODELPORT_PROFILE": "smoke",
        "MODELPORT_S3_ENDPOINT": "http://minio:9000",
        "AWS_ACCESS_KEY_ID": "k",
        "AWS_SECRET_ACCESS_KEY": "s",
        "MLFLOW_TRACKING_URI": "http://mlflow:5000",
        "MODELPORT_RAY_IMAGE": "modelport/ray:test",
    }
)


def registry(champion=True):
    """v1 champion (if any) and v2 the gate-passed candidate."""
    reg = sdk_fakes.InMemoryRegistry()
    for n in (1, 2):
        run = f"train-{n}"
        reg.runs[run] = {"tags": {}, "artifacts": {"spec/model.yaml": SPEC}}
        tags = {
            Tag.TRAIN_RUN_ID: run,
            Tag.RUNTIME_PIP: json.dumps(["xgboost==3.4.1"]),
            Tag.VAL_URI: f"s3://datasets/covertype/v{n}/val.parquet",
        }
        reg.add_version(MODEL, tags, run)
    if champion:
        reg.aliases[MODEL][Alias.CHAMPION] = "1"
        reg.set_version_tags(MODEL, "1", {Tag.STATE: ModelState.CHAMPION.value})
    reg.aliases[MODEL][Alias.CANDIDATE] = "2"
    reg.set_version_tags(MODEL, "2", {Tag.STATE: ModelState.CANDIDATE.value})
    return reg


class Kube:
    """Records every serve config written to the RayService."""

    def __init__(self):
        self.obj = None
        self.configs = []

    def ensure(self, plural, body):
        if self.obj is None:
            self.obj = body
            self.configs.append(yaml.safe_load(body["spec"]["serveConfigV2"]))
            return body, True
        return self.obj, False

    def merge_patch(self, plural, name, patch):
        self.obj["spec"].update(patch["spec"])
        self.configs.append(yaml.safe_load(patch["spec"]["serveConfigV2"]))
        return self.obj


def state(config):
    """(champion version, candidate version, weight) of the rendered config."""
    apps = {a["name"]: a for a in config["applications"]}
    router = apps[MODEL]["deployments"][0]["user_config"]
    version = lambda role: apps.get(f"{MODEL}--{role}", {}).get("args", {}).get("version")  # noqa: E731
    return version("champion"), version("candidate"), router["candidate_weight"]


def start(reg, kube, fault=0.0):
    checked = dag_module.check_candidate(reg, MODEL, "2")
    return checked, dag_module.start_canary(
        reg, kube, SETTINGS, MODEL, "2", checked, fault_error_rate=fault, now=NOW
    )


def test_check_candidate_skips_a_version_that_is_not_the_candidate():
    reg = registry()
    with pytest.raises(AirflowSkipException, match="not the candidate"):
        dag_module.check_candidate(reg, MODEL, "1")
    checked = dag_module.check_candidate(reg, MODEL, "2")
    assert checked == {
        "has_champion": True,
        "window_minutes": 10,
        "canary_weight": 0.1,
        "val_uri": "s3://datasets/covertype/v2/val.parquet",
    }


def test_start_canary_sets_serve_tags():
    reg, kube = registry(), Kube()
    _, started = start(reg, kube, fault=0.5)
    assert reg.model_tags(MODEL)[Tag.SERVE_CANDIDATE_VERSION] == "2"
    assert reg.model_tags(MODEL)[Tag.SERVE_CANDIDATE_WEIGHT] == "0.1"
    assert reg.model_tags(MODEL)[Tag.SERVE_FAULT_ERROR_RATE] == "0.5"
    assert started == {
        "apps": [MODEL, f"{MODEL}--champion", f"{MODEL}--candidate"],
        "started_at": NOW.isoformat(),
        "weight": 0.1,
    }
    assert state(kube.configs[-1]) == ("1", "2", 0.1)


def test_first_model_weight_one():
    reg, kube = registry(champion=False), Kube()
    checked, started = start(reg, kube)
    assert checked["has_champion"] is False
    assert started["weight"] == 1.0
    assert state(kube.configs[-1]) == (None, "2", 1.0)


class Query:
    def __init__(self, values):
        self.values, self.calls = values, []

    def scalar(self, promql, *, at=None):
        self.calls.append(promql)
        app = "candidate" if "--candidate" in promql else "champion"
        kind = "p95" if "quantile" in promql else "errors" if "error" in promql else "requests"
        return self.values[app][kind]


HEALTHY = {
    "candidate": {"requests": 300.0, "errors": 0.0, "p95": 40.0},
    "champion": {"requests": 2700.0, "errors": 0.0, "p95": 42.0},
}


def run_judge(reg, query, deploy_failed=None, has_champion=True):
    return dag_module.judge(
        reg,
        query,
        MODEL,
        "2",
        deploy_failed=deploy_failed,
        has_champion=has_champion,
        window_s=600,
        now=NOW,
        pushgateway_url=None,
    )


def test_judge_passes_and_records_tags():
    reg = registry()
    assert run_judge(reg, Query(HEALTHY)) == {"passed": True, "reasons": []}
    tags = reg.get_version(MODEL, "2").tags
    assert (tags[Tag.CANARY_RESULT], json.loads(tags[Tag.CANARY_REASONS])) == ("passed", [])


def test_judge_deploy_failed_short_circuits_to_rollback():
    reg, query = registry(), Query(HEALTHY)
    verdict = run_judge(reg, query, deploy_failed="covertype-classifier--candidate DEPLOY_FAILED: self-test")
    assert verdict["passed"] is False
    assert verdict["reasons"] == [
        "candidate app did not deploy: covertype-classifier--candidate DEPLOY_FAILED: self-test"
    ]
    assert query.calls == []  # no Prometheus needed
    assert dag_module.branch_choice(verdict) == "remove_candidate_app"


def test_branch_paths():
    assert dag_module.branch_choice({"passed": True}) == "route_all_to_candidate"
    assert dag_module.branch_choice({"passed": False}) == "remove_candidate_app"
    branch = dag_module.DAG.get_task("branch")
    assert set(branch.downstream_task_ids) == {"route_all_to_candidate", "remove_candidate_app"}
    promote = [
        "route_all_to_candidate",
        "wait_route",
        "promote_aliases",
        "update_champion_app",
        "wait_champion_ready",
        "retire_candidate_app",
        "wait_final",
    ]
    for before, after in itertools.pairwise(promote):
        assert after in dag_module.DAG.get_task(before).downstream_task_ids, (before, after)
    rollback = ["remove_candidate_app", "mark_rejected", "wait_after_rollback"]
    for before, after in itertools.pairwise(rollback):
        assert after in dag_module.DAG.get_task(before).downstream_task_ids, (before, after)


def test_promote_path_order():
    reg, kube = registry(), Kube()
    start(reg, kube)
    dagkit.route_all_to_candidate(reg, kube, SETTINGS, MODEL)
    dag_module.promote_aliases(reg, MODEL, "2", None)
    dag_module.apply_now(reg, kube, SETTINGS)
    dagkit.clear_serve_candidate(reg, kube, SETTINGS, MODEL)
    assert [state(c) for c in kube.configs] == [
        ("1", "2", 0.1),  # canary beside the champion
        ("1", "2", 1.0),  # all traffic to the running candidate app; the champion app idles
        ("2", "2", 1.0),  # the idle champion app now runs v2
        ("2", None, 0.0),  # candidate app removed; traffic back on the champion app
    ]
    snapshot = reg.snapshot(MODEL)
    assert (snapshot.champion, snapshot.previous, snapshot.candidate) == ("2", "1", None)


def test_rollback_marks_rejected_and_removes_app():
    reg, kube = registry(), Kube()
    start(reg, kube)
    dagkit.clear_serve_candidate(reg, kube, SETTINGS, MODEL)
    dag_module.mark_rejected(reg, MODEL, "2", None)
    assert state(kube.configs[-1]) == ("1", None, 0.0)
    snapshot = reg.snapshot(MODEL)
    assert (snapshot.champion, snapshot.candidate) == ("1", None)
    assert reg.get_version(MODEL, "2").state is ModelState.REJECTED


class Store:
    def __init__(self, frame):
        buffer = io.BytesIO()
        frame.to_parquet(buffer)
        self.data = buffer.getvalue()

    def read_bytes(self, uri):
        return self.data


def test_canary_traffic_reads_validation_rows(monkeypatch):
    seen = {}

    def fake_load(client, model, rows, *, spec_inputs, rps, duration_s):
        seen.update(model=model, rows=len(rows), rps=rps, duration_s=duration_s, inputs=len(spec_inputs))
        return SimpleNamespace(sent=10, failed=0, status_counts={"200": 10}, variants={"candidate": 10})

    monkeypatch.setattr(dag_module, "run_load", fake_load)
    from modelport.spec import parse_spec

    spec = parse_spec(yaml.safe_load(SPEC))
    result = dag_module.canary_traffic(
        None, Store(pd.DataFrame({"elevation": [1.0, 2.0]})), spec, MODEL, "s3://x/val.parquet", window_s=600
    )
    assert result == {"sent": 10, "failed": 0, "status_counts": {"200": 10}}
    assert seen == {"model": MODEL, "rows": 2, "rps": 5.0, "duration_s": 600, "inputs": 12}
