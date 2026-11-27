import importlib.util
import itertools
import json
import sys
from pathlib import Path

import pytest
import yaml

from modelport.lifecycle import ModelState
from modelport.naming import Alias, Tag

pytest.importorskip("airflow.sdk")
from airflow.sdk.exceptions import AirflowFailException

from modelport_platform import dagkit
from modelport_platform.settings import PlatformSettings

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "dags"))
import serve_sync as dag_module

ROOT = Path(__file__).resolve().parents[3]
SPEC = (ROOT / "examples" / "models" / "covertype-classifier" / "model.yaml").read_bytes()
_spec = importlib.util.spec_from_file_location("sdk_fakes", ROOT / "sdk/tests/fakes.py")
sdk_fakes = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sdk_fakes)
MODEL = "covertype-classifier"
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


def registry(champion="2", previous="1", gate_v3="passed"):
    """v1 and v2 passed the gate; v3's gate result is `gate_v3`."""
    reg = sdk_fakes.InMemoryRegistry()
    for n, gate in ((1, "passed"), (2, "passed"), (3, gate_v3)):
        run = f"train-{n}"
        reg.runs[run] = {"tags": {}, "artifacts": {"spec/model.yaml": SPEC}}
        tags = {Tag.TRAIN_RUN_ID: run, Tag.RUNTIME_PIP: json.dumps([]), Tag.GATE_RESULT: gate}
        reg.add_version(MODEL, tags, run)
    reg.set_version_tags(MODEL, "3", {Tag.STATE: ModelState.CANDIDATE.value})
    if champion:
        reg.aliases[MODEL][Alias.CHAMPION] = champion
        reg.set_version_tags(MODEL, champion, {Tag.STATE: ModelState.CHAMPION.value})
    if previous:
        reg.aliases[MODEL][Alias.PREVIOUS] = previous
        reg.set_version_tags(MODEL, previous, {Tag.STATE: ModelState.PREVIOUS.value})
    reg.set_model_tags(MODEL, {})
    return reg


class Kube:
    def __init__(self):
        self.obj, self.configs = None, []

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
    apps = {a["name"]: a for a in config["applications"]}
    router = apps[MODEL]["deployments"][0]["user_config"]
    version = lambda role: apps.get(f"{MODEL}--{role}", {}).get("args", {}).get("version")  # noqa: E731
    return version("champion"), version("candidate"), router["candidate_weight"]


def test_promote_refused_without_gate_pass():
    with pytest.raises(AirflowFailException, match="v3 did not pass the gate"):
        dag_module.plan(registry(gate_v3="failed"), MODEL, "promote", "3")
    with pytest.raises(AirflowFailException, match="already the champion"):
        dag_module.plan(registry(), MODEL, "promote", "2")
    assert dag_module.plan(registry(), MODEL, "promote", "3") == {"target": "3", "served": True}


def test_promote_refused_during_canary():
    reg = registry()
    reg.set_model_tags(MODEL, {Tag.SERVE_CANDIDATE_VERSION: "3"})
    with pytest.raises(AirflowFailException, match="a canary is running"):
        dag_module.plan(reg, MODEL, "promote", "3")
    reg.set_model_tags(MODEL, {Tag.SERVE_CANDIDATE_VERSION: ""})  # cleared after the canary
    assert dag_module.plan(reg, MODEL, "promote", "3")["target"] == "3"


def test_rollback_refused_without_previous():
    with pytest.raises(AirflowFailException, match="no previous version"):
        dag_module.plan(registry(previous=None), MODEL, "rollback", None)
    assert dag_module.plan(registry(), MODEL, "rollback", None) == {"target": "1", "served": True}


def swap(reg, kube, action, target):
    planned = dag_module.plan(reg, MODEL, action, target)
    assert dag_module.staging_choice(planned) == "stage_target"
    dagkit.serve_target(reg, kube, SETTINGS, MODEL, planned["target"], 0.0)
    dagkit.route_all_to_candidate(reg, kube, SETTINGS, MODEL)
    dag_module.apply_aliases(reg, MODEL, action, planned["target"], None)
    dagkit.apply_serve_state(reg, kube, SETTINGS)
    dagkit.clear_serve_candidate(reg, kube, SETTINGS, MODEL)


def test_rollback_swap_order():
    reg, kube = registry(), Kube()
    swap(reg, kube, "rollback", None)
    assert [state(c) for c in kube.configs] == [
        ("2", "1", 0.0),  # previous staged with no traffic
        ("2", "1", 1.0),  # all traffic to it; the champion app idles
        ("1", "1", 1.0),  # the idle champion app now runs v1
        ("1", None, 0.0),  # second app removed
    ]
    snapshot = reg.snapshot(MODEL)
    assert (snapshot.champion, snapshot.previous) == ("1", None)
    assert reg.get_version(MODEL, "2").state is ModelState.RETIRED


def test_promote_swap_moves_aliases():
    reg, kube = registry(), Kube()
    swap(reg, kube, "promote", "3")
    assert state(kube.configs[-1]) == ("3", None, 0.0)
    snapshot = reg.snapshot(MODEL)
    assert (snapshot.champion, snapshot.previous) == ("3", "2")


def test_promote_unserved_model_skips_staging():
    reg, kube = registry(champion=None, previous=None), Kube()
    planned = dag_module.plan(reg, MODEL, "promote", "3")
    assert planned == {"target": "3", "served": False}
    assert dag_module.staging_choice(planned) == "apply_aliases"
    dag_module.apply_aliases(reg, MODEL, "promote", "3", None)
    dagkit.apply_serve_state(reg, kube, SETTINGS)
    assert [state(c) for c in kube.configs] == [("3", None, 0.0)]


def test_dag_paths():
    branch = dag_module.DAG.get_task("branch")
    assert set(branch.downstream_task_ids) == {"stage_target", "apply_aliases"}
    order = [
        "stage_target",
        "wait_staged",
        "route_all_to_target",
        "wait_routed",
        "apply_aliases",
        "update_champion_app",
        "wait_champion_ready",
        "retire_target_app",
        "wait_final",
    ]
    for before, after in itertools.pairwise(order):
        assert after in dag_module.DAG.get_task(before).downstream_task_ids, (before, after)
    assert dag_module.DAG.get_task("apply_aliases").trigger_rule == "none_failed_min_one_success"
