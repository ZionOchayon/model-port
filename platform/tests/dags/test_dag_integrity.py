"""Every DAG imports cleanly, follows the conventions, and does no I/O at parse time."""

import re
import socket
from pathlib import Path
from unittest import mock

import pytest

airflow_models = pytest.importorskip("airflow.models")

ROOT = Path(__file__).resolve().parents[3]
DAG_FOLDER = ROOT / "platform" / "dags"
EXPECTED = {
    "ingest_dataset",
    "retrain_on_dataset",
    "train_model",
    "evaluate_and_promote",
    "deploy_candidate",
    "serve_sync",
    "batch_inference",
    "drift_check",
    "simulate_drift",
}
NO_RETRY_TYPES = ("TriggerDagRunOperator", "S3KeySensor", "PythonSensor", "DecoratedSensorOperator")


@pytest.fixture(scope="module")
def dagbag():
    return airflow_models.DagBag(dag_folder=str(DAG_FOLDER))


def test_no_import_errors(dagbag):
    assert dagbag.import_errors == {}


def test_expected_dag_ids(dagbag):
    files = {p.stem for p in DAG_FOLDER.glob("*.py")}
    assert set(dagbag.dag_ids) == files  # one DAG per file, named like the file
    assert files <= EXPECTED


def test_every_dag_has_tags_and_retries(dagbag):
    for dag in dagbag.dags.values():
        assert "modelport" in dag.tags, dag.dag_id
        assert dag.catchup is False, dag.dag_id
        for t in dag.tasks:
            if type(t).__name__ not in NO_RETRY_TYPES and not getattr(t, "is_sensor", False):
                assert t.retries >= 1, f"{dag.dag_id}.{t.task_id}"


def test_no_top_level_io():
    def refuse(*args, **kwargs):
        raise AssertionError("DAG files must not do I/O at parse time")

    with mock.patch.object(socket, "socket", refuse), mock.patch("boto3.client", refuse):
        bag = airflow_models.DagBag(dag_folder=str(DAG_FOLDER))
    assert bag.import_errors == {}


def test_pools_referenced_exist_in_up_script(dagbag):
    from modelport_platform.dagkit import Pool

    used = {t.pool for dag in dagbag.dags.values() for t in dag.tasks if t.pool != "default_pool"}
    assert used <= {p.value for p in Pool}
    up = (ROOT / "scripts" / "up.sh").read_text(encoding="utf-8")
    for pool in Pool:
        assert re.search(rf'"{pool.value} ', up), f"up.sh does not create pool {pool.value}"
