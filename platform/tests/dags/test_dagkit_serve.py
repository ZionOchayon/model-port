import importlib.util
import json
from pathlib import Path

import pytest
import yaml

from modelport.naming import Alias, Tag

pytest.importorskip("airflow.sdk")
from airflow.sdk.exceptions import AirflowFailException

from modelport_platform import dagkit
from modelport_platform.kube import Plural
from modelport_platform.settings import PlatformSettings

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


def registry(champion=True, serve_candidate=None, weight="0.1", fault=None):
    reg = sdk_fakes.InMemoryRegistry()
    for n in (1, 2):
        run = f"train-{n}"
        reg.runs[run] = {"tags": {}, "artifacts": {"spec/model.yaml": SPEC}}
        reg.add_version(MODEL, {Tag.TRAIN_RUN_ID: run, Tag.RUNTIME_PIP: json.dumps(["xgboost==3.4.1"])}, run)
    if champion:
        reg.aliases[MODEL][Alias.CHAMPION] = "1"
    tags = {}
    if serve_candidate:
        tags = {Tag.SERVE_CANDIDATE_VERSION: serve_candidate, Tag.SERVE_CANDIDATE_WEIGHT: weight}
        if fault is not None:
            tags[Tag.SERVE_FAULT_ERROR_RATE] = fault
    reg.set_model_tags(MODEL, tags)
    reg.ensure_registered_model("never-served", {})
    return reg


def test_champion_only():
    (model,) = dagkit.desired_serve_state(registry())
    assert (model.name, model.champion.version, model.candidate, model.candidate_weight) == (
        MODEL,
        "1",
        None,
        0.0,
    )
    assert model.champion.runtime_pip == ("xgboost==3.4.1",)


def test_serve_candidate_with_weight_and_fault():
    (model,) = dagkit.desired_serve_state(registry(serve_candidate="2", weight="0.1", fault="1.0"))
    assert (model.champion.version, model.candidate.version, model.candidate_weight) == ("1", "2", 0.1)
    assert (model.champion.fault_error_rate, model.candidate.fault_error_rate) == (0.0, 1.0)


def test_candidate_alias_alone_is_not_served():
    reg = registry(champion=False)
    reg.aliases[MODEL][Alias.CANDIDATE] = "2"  # passed the gate, canary not started
    assert dagkit.desired_serve_state(reg) == []


def test_first_version_served_as_candidate_only():
    (model,) = dagkit.desired_serve_state(registry(champion=False, serve_candidate="1", weight="0.1"))
    assert (model.champion, model.candidate.version, model.candidate_weight) == (None, "1", 1.0)


class FakeKube:
    def __init__(self, existing=None):
        self.objects = {} if existing is None else {existing["metadata"]["name"]: existing}
        self.patches = []

    def ensure(self, plural, body):
        assert plural == Plural.RAYSERVICES
        name = body["metadata"]["name"]
        if name in self.objects:
            return self.objects[name], False
        self.objects[name] = body
        return body, True

    def get(self, plural, name):
        return self.objects.get(name)

    def merge_patch(self, plural, name, patch):
        self.patches.append(patch)
        self.objects[name]["spec"].update(patch["spec"])
        return self.objects[name]


def test_apply_creates_then_patches_serve_config():
    kube = FakeKube()
    apps = dagkit.apply_serve_state(registry(), kube, SETTINGS)
    assert apps == [MODEL, f"{MODEL}--champion"]
    assert kube.patches == []
    apps = dagkit.apply_serve_state(registry(serve_candidate="2"), kube, SETTINGS)
    assert apps == [MODEL, f"{MODEL}--champion", f"{MODEL}--candidate"]
    (patch,) = kube.patches
    assert patch == {"spec": {"serveConfigV2": patch["spec"]["serveConfigV2"]}}  # only the serve config
    rendered = yaml.safe_load(kube.objects["modelport-serve"]["spec"]["serveConfigV2"])
    assert [a["name"] for a in rendered["applications"]] == apps


def status(apps, generation=1, observed=1):
    return {
        "metadata": {"name": "modelport-serve", "generation": generation},
        "spec": {},
        "status": {
            "observedGeneration": observed,
            "serviceStatus": "Running",
            "activeServiceStatus": {"applicationStatuses": apps},
        },
    }


def test_ready_poke():
    apps = ["m", "m--champion"]
    assert dagkit.rayservice_ready_poke(FakeKube(), apps, name="modelport-serve").is_done is False
    running = {a: {"status": "RUNNING"} for a in apps}
    assert dagkit.rayservice_ready_poke(FakeKube(status(running)), apps, name="modelport-serve").is_done
    stale = status(running, generation=2, observed=1)
    assert dagkit.rayservice_ready_poke(FakeKube(stale), apps, name="modelport-serve").is_done is False
    failed = {**running, "m--champion": {"status": "DEPLOY_FAILED", "message": "self-test failed"}}
    with pytest.raises(AirflowFailException, match="m--champion DEPLOY_FAILED"):
        dagkit.rayservice_ready_poke(FakeKube(status(failed)), apps, name="modelport-serve")
