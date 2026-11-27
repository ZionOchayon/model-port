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

    def get_run(self, dag_id, run_id):
        from modelport.clients.airflow import DagRunInfo

        if not any(t[1] == run_id for t in self.triggered):
            return None
        return DagRunInfo(dag_id, run_id, "success", None, None, {}, f"http://airflow/{run_id}")

    def wait(self, dag_id, run_id, *, timeout_s, poll_s=10.0):
        from modelport.clients.airflow import DagRunInfo

        self.waited.append((dag_id, run_id))
        return DagRunInfo(dag_id, run_id, "success", None, None, {}, f"http://airflow/{run_id}")


@pytest.fixture
def services(monkeypatch, memory_registry):
    orchestrator = FakeOrchestrator()
    settings = main.ClientSettings.from_env(
        {"MODELPORT_AIRFLOW_PASSWORD": "pw"}, credentials_file=Path("/none")
    )
    svc = main.Services(
        registry=memory_registry, store=memory_registry, orchestrator=orchestrator, settings=settings
    )
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
    key = services.registry.runs[submission]["tags"][Tag.SUBMISSION_KEY]
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


def _gate_report(version, metric, per_class):
    return {
        "model": "toy-model",
        "version": version,
        "val_uri": "s3://datasets/toy/v1/val.parquet",
        "n_samples": 100,
        "metric_name": "f1_macro",
        "metric": metric,
        "per_class": per_class,
        "latency_ms_median": 4.2,
        "latency_batch_size": 256,
        "created_at": "2026-11-27T10:00:00Z",
    }


@pytest.fixture
def decided_model(services):
    from modelport.naming import Alias, Tag

    reg = services.registry
    reg.ensure_registered_model("toy-model", {})
    reg.add_version("toy-model", {Tag.STATE: "champion"})
    reg.add_version(
        "toy-model",
        {
            Tag.STATE: "candidate",
            Tag.GATE_RESULT: "passed",
            Tag.GATE_REASONS: "[]",
            Tag.GATE_EVAL_RUN_ID: "eval-2",
        },
    )
    reg.aliases["toy-model"].update({Alias.CHAMPION: "1", Alias.CANDIDATE: "2"})
    reg.log_json("eval-2", "gate/report.json", _gate_report("2", 0.91, {"a": 0.9, "b": 0.92}))
    reg.log_json("eval-2", "gate/champion_report.json", _gate_report("1", 0.85, {"a": 0.84, "b": 0.86}))
    return services


def test_status_shows_aliases_gate_and_links(decided_model):
    result = run("status", "toy-model")
    assert result.exit_code == 0, result.output
    out = result.output
    assert "champion v1" in out
    assert "candidate v2" in out
    assert "previous -" in out
    assert "newest v2: candidate, gate passed" in out
    assert "0.910" in out  # the version's metric
    assert "0.850" in out  # the champion's metric
    assert "http://localhost:5000/#/models/toy-model" in out
    assert "http://localhost:8265/#/jobs" in out
    assert "evaluate__toy-model__v2" in out


def test_status_json(decided_model):
    result = run("--json", "status", "toy-model")
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["aliases"] == {"champion": "1", "previous": None, "candidate": "2"}
    assert payload["newest"]["gate"]["report"]["metric"] == 0.91
    assert payload["newest"]["gate"]["champion_report"]["metric"] == 0.85


def test_status_rejected_shows_reasons(services):
    from modelport.naming import Tag

    services.registry.ensure_registered_model("toy-model", {})
    services.registry.add_version(
        "toy-model",
        {Tag.STATE: "rejected", Tag.GATE_RESULT: "failed", Tag.GATE_REASONS: '["f1_macro 0.4 < 0.5"]'},
    )
    result = run("status", "toy-model")
    assert result.exit_code == 0, result.output
    assert "newest v1: rejected, gate failed" in result.output
    assert "f1_macro 0.4 < 0.5" in result.output


def test_status_unknown_model_exit_7(services):
    result = run("status", "nope-model")
    assert result.exit_code == 7
    assert "No model named 'nope-model'" in result.output


def test_submit_json_is_one_document(services, monkeypatch):
    import os

    monkeypatch.delenv("MLFLOW_SUPPRESS_PRINTING_URL_TO_STDOUT", raising=False)
    result = run("--json", "submit", str(FIXTURES / "valid_model"))
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["train_run_id"].startswith("submit__toy-model__")
    # MLflow would otherwise print "View run ..." lines into stdout and break --json
    assert os.environ["MLFLOW_SUPPRESS_PRINTING_URL_TO_STDOUT"] == "true"


def served(registry, *, gate_v2="passed", previous=True):
    from modelport.naming import Alias, Tag

    registry.add_version(
        "tiny-model", {Tag.GATE_RESULT: "passed", Tag.STATE: "champion" if not previous else "previous"}
    )
    registry.add_version("tiny-model", {Tag.GATE_RESULT: "passed", Tag.STATE: "champion"})
    registry.add_version(
        "tiny-model",
        {Tag.GATE_RESULT: gate_v2, Tag.STATE: "candidate" if gate_v2 == "passed" else "rejected"},
    )
    registry.aliases["tiny-model"][Alias.CHAMPION] = "2"
    if previous:
        registry.aliases["tiny-model"][Alias.PREVIOUS] = "1"


def test_promote_refused_when_gate_failed_exit_8(services):
    served(services.registry, gate_v2="failed")
    result = run("promote", "tiny-model", "3", "--yes")
    assert result.exit_code == 8
    assert "did not pass the gate" in result.output
    assert services.orchestrator.triggered == []


def test_promote_triggers_serve_sync(services):
    served(services.registry)
    result = run("--json", "promote", "tiny-model", "3", "--yes", "--wait")
    assert result.exit_code == 0, result.output
    ((dag_id, run_id, conf),) = services.orchestrator.triggered
    assert dag_id == "serve_sync"
    assert run_id.startswith("promote__tiny-model__")
    assert conf == {"model": "tiny-model", "action": "promote", "version": "3"}
    assert services.orchestrator.waited == [("serve_sync", run_id)]
    payload = json.loads(result.output)
    assert (payload["action"], payload["target"]) == ("promote", "3")


def test_promote_asks_for_confirmation(services):
    served(services.registry)
    result = runner.invoke(main.app, ["promote", "tiny-model", "3"], input="n\n")
    assert result.exit_code == 1  # aborted
    assert services.orchestrator.triggered == []


def test_rollback_without_previous_exit_8(services):
    served(services.registry, previous=False)
    result = run("rollback", "tiny-model", "--yes")
    assert result.exit_code == 8
    assert "no previous version" in result.output


def test_rollback_triggers_serve_sync(services):
    served(services.registry)
    result = run("rollback", "tiny-model", "--yes")
    assert result.exit_code == 0, result.output
    ((dag_id, run_id, conf),) = services.orchestrator.triggered
    assert (dag_id, conf) == ("serve_sync", {"model": "tiny-model", "action": "rollback", "version": None})
    assert run_id.startswith("rollback__tiny-model__")
    assert "v1" in result.output
