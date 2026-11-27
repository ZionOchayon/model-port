"""MlflowRegistry against a real MLflow (sqlite backend, local artifacts)."""

import pytest

from modelport.lifecycle import AliasSnapshot, Event, ModelState, plan_gate, plan_promote, plan_rollback
from modelport.naming import Alias, Tag
from modelport.packaging import build_code_zip
from modelport.registry import MlflowRegistry

pytestmark = pytest.mark.integration


@pytest.fixture
def registry(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    return MlflowRegistry(f"sqlite:///{tmp_path / 'mlflow.db'}", sleep=lambda s: None)


def version(registry, name):
    """Create a bare model version (no artifacts needed for alias logic)."""
    client = registry.client
    run = client.create_run(client.create_experiment(f"e-{name}-{len(client.search_experiments())}"))
    mv = client.create_model_version(name, source=f"{run.info.artifact_uri}/model", run_id=run.info.run_id)
    return str(mv.version)  # MLflow returns an int; modelport's contract uses strings


def test_ensure_registered_model_idempotent(registry):
    registry.ensure_registered_model("image-classifier", {Tag.DATASET: "fashion-mnist"})
    registry.ensure_registered_model("image-classifier", {Tag.DATASET: "eurosat", Tag.AUTO_RETRAIN: "true"})
    tags = registry.model_tags("image-classifier")
    assert tags[Tag.DATASET] == "eurosat"
    assert tags[Tag.AUTO_RETRAIN] == "true"


def test_resolve_missing_alias_returns_none(registry):
    registry.ensure_registered_model("image-classifier", {})
    assert registry.resolve("image-classifier", Alias.CHAMPION) is None


def test_apply_promote_plan_sets_aliases_and_states(registry):
    registry.ensure_registered_model("image-classifier", {})
    v1, v2 = version(registry, "image-classifier"), version(registry, "image-classifier")
    registry.apply_plan("image-classifier", plan_gate(registry.snapshot("image-classifier"), v1, passed=True))
    snap = registry.snapshot("image-classifier")
    registry.apply_plan(
        "image-classifier",
        plan_promote(snap, v1, event=Event.CANARY_PASSED, current_state=ModelState.CANDIDATE),
    )
    registry.apply_plan("image-classifier", plan_gate(registry.snapshot("image-classifier"), v2, passed=True))
    snap = registry.snapshot("image-classifier")
    assert (snap.champion, snap.candidate, snap.previous) == (v1, v2, None)
    registry.apply_plan(
        "image-classifier",
        plan_promote(snap, v2, event=Event.CANARY_PASSED, current_state=ModelState.CANDIDATE),
    )
    snap = registry.snapshot("image-classifier")
    assert (snap.champion, snap.previous, snap.candidate) == (v2, v1, None)
    assert snap.states == {v1: ModelState.PREVIOUS, v2: ModelState.CHAMPION}
    info = registry.get_version("image-classifier", v2)
    assert info.state is ModelState.CHAMPION
    assert info.tags[Tag.STATE_CHANGED_AT].endswith("Z")
    assert registry.resolve("image-classifier", Alias.CHAMPION).version == v2


def test_apply_plan_twice_is_noop(registry):
    registry.ensure_registered_model("image-classifier", {})
    v1 = version(registry, "image-classifier")
    plan = plan_gate(registry.snapshot("image-classifier"), v1, passed=True)
    registry.apply_plan("image-classifier", plan)
    first = registry.snapshot("image-classifier")
    registry.apply_plan("image-classifier", plan)
    assert registry.snapshot("image-classifier") == first


def test_rollback_plan_roundtrip(registry):
    registry.ensure_registered_model("image-classifier", {})
    v1, v2 = version(registry, "image-classifier"), version(registry, "image-classifier")
    for v in (v1, v2):
        registry.apply_plan(
            "image-classifier", plan_gate(registry.snapshot("image-classifier"), v, passed=True)
        )
        snap = registry.snapshot("image-classifier")
        registry.apply_plan(
            "image-classifier",
            plan_promote(snap, v, event=Event.CANARY_PASSED, current_state=ModelState.CANDIDATE),
        )
    registry.apply_plan("image-classifier", plan_rollback(registry.snapshot("image-classifier")))
    snap = registry.snapshot("image-classifier")
    assert snap == AliasSnapshot(
        model="image-classifier",
        champion=v1,
        previous=None,
        candidate=None,
        states={v1: ModelState.CHAMPION},
    )
    assert registry.get_version("image-classifier", v2).state is ModelState.RETIRED


def test_find_submission_by_key(registry, tmp_path):
    model_dir = tmp_path / "m"
    model_dir.mkdir()
    (model_dir / "model.py").write_text("x = 1\n", encoding="utf-8")
    spec = model_dir / "model.fashion.yaml"
    spec.write_text("name: m\n", encoding="utf-8")
    package = build_code_zip(model_dir, tmp_path / "out")
    assert registry.find_submission("image-classifier", "k" * 64) is None
    run_id = registry.create_submission("image-classifier", package, spec, "5" * 64)
    from modelport.naming import submission_key

    key = submission_key(package.sha256, "model.fashion.yaml", "5" * 64)
    assert registry.find_submission("image-classifier", key) == run_id
    assert registry.find_submission("other-model", key) is None
    run = registry.client.get_run(run_id)
    assert run.info.status == "FINISHED"
    assert run.data.tags[Tag.CODE_SHA256] == package.sha256
    files = {a.path for a in registry.client.list_artifacts(run_id, "code")}
    assert files == {"code/code.zip"}
    assert (registry.download(run_id, "spec/model.fashion.yaml", tmp_path / "dl")).read_text() == "name: m\n"


def test_log_and_read_json(registry):
    client = registry.client
    run_id = client.create_run(client.create_experiment("json")).info.run_id
    registry.log_json(run_id, "gate/report.json", {"metric": 0.9, "per_class": {"a": 1.0}})
    registry.log_json(run_id, "gate/report.json", {"metric": 0.95})
    assert registry.read_json(run_id, "gate/report.json") == {"metric": 0.95}


def test_find_version_by_tag_and_list_models(registry):
    registry.ensure_registered_model(
        "image-classifier", {Tag.DATASET: "fashion-mnist", Tag.AUTO_RETRAIN: "true"}
    )
    registry.ensure_registered_model(
        "covertype-classifier", {Tag.DATASET: "covertype", Tag.AUTO_RETRAIN: "true"}
    )
    registry.ensure_registered_model(
        "frozen-model", {Tag.DATASET: "fashion-mnist", Tag.AUTO_RETRAIN: "false"}
    )
    v1 = version(registry, "image-classifier")
    registry.set_version_tags("image-classifier", v1, {Tag.TRAIN_RAYJOB: "train-image-classifier-ab12cd34"})
    assert (
        registry.find_version_by_tag(
            "image-classifier", Tag.TRAIN_RAYJOB, "train-image-classifier-ab12cd34"
        ).version
        == v1
    )
    assert registry.find_version_by_tag("image-classifier", Tag.TRAIN_RAYJOB, "nope") is None
    assert registry.list_models(tag_equals={Tag.DATASET: "fashion-mnist", Tag.AUTO_RETRAIN: "true"}) == [
        "image-classifier"
    ]
    assert registry.list_models() == ["covertype-classifier", "frozen-model", "image-classifier"]
    assert registry.model_uri("image-classifier", v1) == f"models:/image-classifier/{v1}"


def test_ensure_run_finds_by_key_tag_and_keeps_first_params(registry):
    tags = {Tag.TRAIN_RAYJOB: "train-m-1", Tag.MODEL: "m"}
    first = registry.ensure_run("modelport-train/m", "train-m-1", Tag.TRAIN_RAYJOB, tags, {"epochs": 2})
    again = registry.ensure_run("modelport-train/m", "train-m-1", Tag.TRAIN_RAYJOB, tags, {"epochs": 3})
    other = registry.ensure_run(
        "modelport-train/m", "train-m-2", Tag.TRAIN_RAYJOB, {**tags, Tag.TRAIN_RAYJOB: "train-m-2"}, {}
    )
    assert first == again != other
    run = registry.client.get_run(first)
    assert run.data.params == {"epochs": "2"}
    assert run.data.tags[Tag.MODEL] == "m"


def test_run_tags(registry):
    run_id = registry.ensure_run(
        "e", "r", Tag.TRAIN_RAYJOB, {Tag.TRAIN_RAYJOB: "r", Tag.CODE_SHA256: "c"}, {}
    )
    assert registry.run_tags(run_id)[Tag.CODE_SHA256] == "c"


def test_latest_version(registry):
    registry.ensure_registered_model("image-classifier", {})
    assert registry.latest_version("image-classifier") is None
    versions = [version(registry, "image-classifier") for _ in range(11)]
    assert registry.latest_version("image-classifier").version == versions[-1] == "11"  # numeric, not "9"
