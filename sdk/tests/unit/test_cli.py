import json
from pathlib import Path

import pytest
import typer
from typer.testing import CliRunner

from modelport.cli import main

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "broken_models"
runner = CliRunner()


def run(*args):
    return runner.invoke(main.app, list(args))


def test_validate_ok_exit_0():
    result = run("validate", str(FIXTURES / "valid_model"))
    assert result.exit_code == 0, result.output
    assert "✔ model.predict" in result.output


@pytest.mark.parametrize(
    ("fixture", "code"),
    [
        ("missing_yaml", 2),
        ("unpinned_pip", 2),
        ("not_subclass", 3),
        ("example_wrong_type", 4),
        ("predict_raises", 4),
    ],
)
def test_validate_broken_exit_code_and_fix_printed(fixture, code):
    result = run("validate", str(FIXTURES / fixture))
    assert result.exit_code == code, result.output
    assert "✖" in result.output
    assert "How to fix:" in result.output


def test_validate_json_output():
    result = run("--json", "validate", str(FIXTURES / "no_train_loop"))
    payload = json.loads(result.output)
    assert result.exit_code == 3
    assert payload["ok"] is False
    assert payload["first_failure"]["name"] == "class.train_loop"


def test_version_flag():
    result = run("--version")
    assert result.exit_code == 0
    assert result.output.startswith("modelport ")


def test_unexpected_error_exit_1_no_traceback_without_debug(monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("disk on fire")

    monkeypatch.setattr(main, "validate_model_dir", boom)
    result = run("validate", str(FIXTURES / "valid_model"))
    assert result.exit_code == 1
    assert "Unexpected error: RuntimeError: disk on fire" in result.output
    assert "re-run with --debug" in result.output
    assert "Traceback" not in result.output
    debug = run("--debug", "validate", str(FIXTURES / "valid_model"))
    assert debug.exit_code == 1
    assert "Traceback" in debug.output


def test_help_texts_present():
    command = typer.main.get_command(main.app)
    assert command.help
    for param in command.params:
        assert param.help, param.name
    for name, sub in command.commands.items():
        assert sub.help, name
        for param in sub.params:
            assert getattr(param, "help", None), f"{name} {param.name}"
    assert run("--help").exit_code == 0


class FakeOrchestrator:
    def __init__(self, dags=("train_model",)):
        self.dags, self.triggered, self.waited = set(dags), [], []

    def trigger(self, dag_id, run_id, conf):
        from modelport.clients.airflow import DagRunRef

        existed = any(t[1] == run_id for t in self.triggered)
        self.triggered.append((dag_id, run_id, dict(conf)))
        return DagRunRef(dag_id, run_id, existed, f"http://airflow/dags/{dag_id}/runs/{run_id}")

    def has_dag(self, dag_id):
        return dag_id in self.dags

    def wait(self, dag_id, run_id, *, timeout_s, poll_s=10.0):
        from modelport.clients.airflow import DagRunInfo

        self.waited.append((dag_id, run_id))
        return DagRunInfo(dag_id, run_id, "success", None, None, {}, f"http://airflow/{run_id}")


@pytest.fixture
def services(monkeypatch, memory_registry):
    orchestrator = FakeOrchestrator()
    svc = main.Services(registry=memory_registry, store=memory_registry, orchestrator=orchestrator)
    monkeypatch.setattr(main, "build_services", lambda: svc)
    return svc


def submit(*extra):
    return run("submit", str(FIXTURES / "valid_model"), *extra)


def test_submit_triggers_train_with_deterministic_run_id(services):
    from modelport.naming import Tag, submit_run_id

    result = submit("--dataset-version", "3")
    assert result.exit_code == 0, result.output
    ((dag_id, run_id, conf),) = services.orchestrator.triggered
    (submission,) = services.registry.runs
    key = services.registry.runs[submission]["tags"]["key"]
    assert (dag_id, run_id) == ("train_model", submit_run_id("toy-model", key))
    assert conf == {
        "model": "toy-model",
        "submission_run_id": submission,
        "dataset_version": 3,
        "spec_file": "model.yaml",
    }
    tags = services.registry.model_tags("toy-model")
    assert tags[Tag.DATASET] == "toy"
    assert tags[Tag.AUTO_RETRAIN] == "true"
    assert tags[Tag.LATEST_SUBMISSION_RUN_ID] == submission
    assert "Submitted toy-model (code " in result.output
    assert "Next: modelport status toy-model" in result.output


def test_submit_twice_reuses_submission_and_reports_existing(services):
    assert submit().exit_code == 0
    again = submit()
    assert again.exit_code == 0, again.output
    assert len(services.registry.runs) == 1
    assert services.orchestrator.triggered[0][1] == services.orchestrator.triggered[1][1]
    assert "This exact code was already submitted; showing the existing run." in again.output


def test_submit_invalid_model_uploads_nothing(services):
    result = run("submit", str(FIXTURES / "unpinned_pip"))
    assert result.exit_code == 2
    assert "How to fix:" in result.output
    assert services.registry.runs == {}
    assert services.orchestrator.triggered == []


def test_submit_wait_stops_when_evaluation_dag_missing(services):
    result = submit("--wait", "--timeout", "60")
    assert result.exit_code == 0, result.output
    assert services.orchestrator.waited == [("train_model", services.orchestrator.triggered[0][1])]
    assert "training finished; evaluation DAG not installed" in result.output


def _wait_with_gate(services, gate_tags):
    from modelport.naming import Tag, rayjob_name

    services.orchestrator.dags.add("evaluate_and_promote")
    original = services.orchestrator.wait

    def wait(dag_id, run_id, **kw):
        if dag_id == "train_model":  # the training run registered a version from its RayJob
            tags = {Tag.TRAIN_RAYJOB: rayjob_name("toy-model", run_id), **gate_tags}
            services.registry.add_version("toy-model", tags)
        return original(dag_id, run_id, **kw)

    services.orchestrator.wait = wait
    return submit("--wait")


def test_submit_wait_candidate(services):
    from modelport.naming import Tag

    result = _wait_with_gate(services, {Tag.GATE_RESULT: "passed"})
    assert result.exit_code == 0, result.output
    assert services.orchestrator.waited[-1] == ("evaluate_and_promote", "evaluate__toy-model__v1")
    assert "candidate" in result.output


def test_submit_wait_rejected_exit_8(services):
    from modelport.naming import Tag

    result = _wait_with_gate(
        services, {Tag.GATE_RESULT: "failed", Tag.GATE_REASONS: '["macro_f1 0.41 < 0.70"]'}
    )
    assert result.exit_code == 8, result.output
    assert "rejected: macro_f1 0.41 < 0.70" in result.output
