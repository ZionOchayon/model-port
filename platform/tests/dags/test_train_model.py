import importlib.util
import json
import sys
from pathlib import Path

import pytest

from modelport.errors import SpecError
from modelport.naming import Tag, rayjob_name

pytest.importorskip("airflow.sdk")
from airflow.sdk.exceptions import AirflowFailException, ParamValidationError

from modelport_platform import dagkit

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "dags"))
import train_model as dag_module

ROOT = Path(__file__).resolve().parents[3]
SPEC = ROOT / "examples" / "models" / "image-classifier" / "model.fashion.yaml"
_spec = importlib.util.spec_from_file_location("sdk_fakes", ROOT / "sdk/tests/fakes.py")
sdk_fakes = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sdk_fakes)


class Store:
    def __init__(self, files):
        self.files = files

    def read_bytes(self, uri):
        return self.files[uri]

    def exists(self, uri):
        return uri in self.files

    def list(self, prefix_uri):
        return sorted(u for u in self.files if u.startswith(prefix_uri))


def registry():
    reg = sdk_fakes.InMemoryRegistry()
    reg.runs["run-1"] = {"tags": {}, "artifacts": {"spec/model.fashion.yaml": SPEC.read_bytes()}}
    return reg


def store(versions=(1,)):
    manifest = {"rows_train": 90, "rows_val": 10, "labels": ["Bag", "Coat"], "sha256": "ab"}
    return Store(
        {
            f"s3://datasets/fashion-mnist/v{v}/manifest.json": json.dumps({**manifest, "version": v}).encode()
            for v in versions
        }
    )


CONF = {
    "model": "fashion-classifier",
    "submission_run_id": "run-1",
    "dataset_version": None,
    "spec_file": "model.fashion.yaml",
}


def test_prepare_builds_names_and_uri(tmp_path):
    out = dag_module.prepare(CONF, "manual__2026-11-27", registry(), store((1, 2)), tmp_path)
    assert out["rayjob"] == rayjob_name("fashion-classifier", "manual__2026-11-27")
    assert out["working_dir_uri"] == "s3://mlflow/0/run-1/artifacts/code/code.zip"
    assert out["dataset_version"] == 2
    assert json.loads(out["spec_json"])["name"] == "fashion-classifier"
    pinned = dag_module.prepare({**CONF, "dataset_version": 1}, "r", registry(), store((1, 2)), tmp_path)
    assert pinned["dataset_version"] == 1


def test_prepare_rejects_spec_for_other_model(tmp_path):
    with pytest.raises(SpecError, match="other-model"):
        dag_module.prepare({**CONF, "model": "other-model"}, "r", registry(), store(), tmp_path)


def test_prepare_without_dataset_version_fails(tmp_path):
    with pytest.raises(SpecError, match="has no versions yet"):
        dag_module.prepare(CONF, "r", registry(), store(()), tmp_path)


class Kube:
    def __init__(self, obj):
        self.obj = obj

    def get(self, plural, name):
        assert (plural, name) == ("rayjobs", "train-x")
        return self.obj


def rayjob(deployment, job=None, **status):
    return {
        "metadata": {"name": "train-x", "creationTimestamp": "2026-11-27T10:00:00Z"},
        "status": {"jobDeploymentStatus": deployment, "jobStatus": job, **status},
    }


def test_rayjob_poke_states():
    assert dagkit.rayjob_poke(Kube(rayjob("Initializing")), "train-x").is_done is False
    assert dagkit.rayjob_poke(Kube(rayjob("Running", "RUNNING")), "train-x").is_done is False
    done = dagkit.rayjob_poke(
        Kube(
            rayjob(
                "Complete",
                "SUCCEEDED",
                startTime="2026-11-27T10:00:30Z",
                endTime="2026-11-27T10:05:30Z",
            )
        ),
        "train-x",
    )
    assert done.is_done is True
    assert done.xcom_value == {"startup_s": None, "duration_s": 300.0}
    with pytest.raises(AirflowFailException, match=r"kubectl -n ray logs job/train-x"):
        dagkit.rayjob_poke(Kube(rayjob("Failed", "FAILED", reason="AppFailed", message="exit 1")), "train-x")
    gone = dagkit.rayjob_poke(Kube(None), "train-x")  # deleted by its TTL: record_version decides
    assert gone.is_done is True


def test_record_version_missing_fails():
    reg = registry()
    with pytest.raises(AirflowFailException, match="training finished but registered no version"):
        dag_module.record_version(reg, "fashion-classifier", "train-x")
    reg.ensure_registered_model("fashion-classifier", {})
    reg.add_version("fashion-classifier", {Tag.TRAIN_RAYJOB: "train-x"})
    assert dag_module.record_version(reg, "fashion-classifier", "train-x") == "1"


def test_params_reject_bad_model_name():
    params = dag_module.DAG.params
    with pytest.raises(ParamValidationError):
        params.get_param("model").resolve("Bad_Name")
    with pytest.raises(ParamValidationError):
        params.get_param("spec_file").resolve("../model.yaml")
    assert params.get_param("model").resolve("fashion-classifier") == "fashion-classifier"


def test_task_ids_and_pools():
    tasks = {t.task_id: t for t in dag_module.DAG.tasks}
    assert set(tasks) == {"prepare", "create_rayjob", "wait_rayjob", "record_version", "trigger_evaluation"}
    trigger = tasks["trigger_evaluation"]
    assert trigger.trigger_dag_id == "evaluate_and_promote"
    assert "evaluate__{{ params.model }}__v" in trigger.trigger_run_id  # naming.evaluate_run_id format
    assert tasks["create_rayjob"].pool == tasks["wait_rayjob"].pool == "ray_training"
    assert tasks["wait_rayjob"].mode == "reschedule"
    assert dag_module.DAG.max_active_runs == 2
