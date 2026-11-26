import subprocess
import sys
from pathlib import Path

import pytest

from modelport.validate import CheckStatus, validate_model_dir

ROOT = Path(__file__).resolve().parents[3]
FIXTURES = ROOT / "sdk" / "tests" / "fixtures" / "broken_models"
EXAMPLES = sorted(p for p in (ROOT / "examples" / "models").glob("*") if (p / "model.yaml").exists())

CASES = [
    ("missing_yaml", "spec.file"),
    ("unpinned_pip", "spec.schema"),
    ("unknown_field", "spec.schema"),
    ("class_not_found", "class.import"),
    ("not_subclass", "class.subclass"),
    ("missing_predict", "class.methods"),
    ("preprocess_two_args", "class.methods"),
    ("no_train_loop", "class.train_loop"),
    ("example_missing_column", "example.input"),
    ("example_wrong_type", "example.input"),
    ("predict_wrong_length", "model.predict"),
    ("predict_bad_confidence", "model.predict"),
    ("predict_unknown_label", "model.predict"),
    ("predict_raises", "model.predict"),
    ("input_stats_wrong_rows", "model.input_stats"),
]


@pytest.mark.parametrize(("fixture", "check"), CASES, ids=[f"test_{c[0]}" for c in CASES])
def test_broken_fixture_fails_expected_check(fixture, check):
    report = validate_model_dir(FIXTURES / fixture)
    failure = report.first_failure()
    assert not report.ok
    assert failure is not None
    assert failure.name == check
    assert failure.fix


def test_structural_failure_skips_the_rest():
    report = validate_model_dir(FIXTURES / "not_subclass")
    statuses = {c.name: c.status for c in report.checks}
    assert statuses["class.subclass"] is CheckStatus.FAILED
    assert {statuses[n] for n in ("class.methods", "example.input", "model.predict")} == {CheckStatus.SKIPPED}


def test_valid_fixture_passes_with_all_checks_in_order():
    report = validate_model_dir(FIXTURES / "valid_model")
    assert report.ok
    assert [c.name for c in report.checks] == [
        "dir.exists",
        "spec.file",
        "spec.schema",
        "class.import",
        "class.subclass",
        "class.methods",
        "class.train_loop",
        "example.input",
        "model.load",
        "model.predict",
        "model.input_stats",
        "model.input_stats.coverage",
    ]
    assert report.spec is not None


def test_no_predict_skips_model_checks():
    report = validate_model_dir(FIXTURES / "predict_raises", run_predict=False)
    assert report.ok
    assert {c.status for c in report.checks if c.name.startswith("model.")} == {CheckStatus.SKIPPED}


def test_missing_dir():
    report = validate_model_dir(FIXTURES / "does-not-exist")
    assert report.first_failure().name == "dir.exists"


def test_no_train_loop_message():
    failure = validate_model_dir(FIXTURES / "no_train_loop").first_failure()
    assert failure.message == "Toy does not implement train_loop"
    assert "docs/components/sdk.md#train_loop" in failure.fix


@pytest.mark.parametrize("model_dir", EXAMPLES, ids=lambda p: p.name)
def test_valid_examples_pass(model_dir):
    assert validate_model_dir(model_dir).ok


def test_no_ray_import():
    code = (
        "import sys; sys.modules['ray'] = None\n"
        "from pathlib import Path\n"
        "from modelport.validate import validate_model_dir\n"
        f"report = validate_model_dir(Path({str(FIXTURES / 'valid_model')!r}))\n"
        "assert report.ok, report.first_failure()\n"
        "assert 'ray' not in {m.split('.')[0] for m in sys.modules if sys.modules[m] is not None}\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
