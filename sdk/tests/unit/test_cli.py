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
