import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
import check_versions


@pytest.mark.parametrize(
    ("version", "image", "ok"),
    [
        ("2.60.1", "rayproject/ray:2.60.1-py312-cpu@sha256:ab", True),
        ("3.18.0", "ghcr.io/mlflow/mlflow:v3.18.0-full@sha256:ab", True),
        ("v3.16.0", "quay.io/prometheus/prometheus:v3.16.0@sha256:ab", True),
        ("2.60.1", "rayproject/ray:2.60.10-py312-cpu@sha256:ab", False),
        ("2.60.1", "rayproject/ray:2.60.0-py312-cpu@sha256:ab", False),
    ],
)
def test_tag_matches(version, image, ok):
    assert check_versions.tag_matches(version, image) is ok


def test_compare_reports_wrong_and_missing_pins():
    assert check_versions.compare("r", ["1.0", "1.0"], "1.0").ok
    wrong = check_versions.compare("r", ["1.0", "0.9"], "1.0")
    assert not wrong.ok
    assert "0.9" in wrong.detail
    assert not check_versions.compare("r", [], "1.0").ok


def test_missing_file_is_skipped_not_failed():
    (result,) = check_versions.file_pin_results({"K": "1"}, "K", "does/not/exist.txt", r"(x)")
    assert result.ok
    assert result.skipped


def test_repository_pins_agree():
    assert check_versions.main([]) == 0
