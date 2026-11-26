import tomllib
from pathlib import Path

import modelport


def test_version_matches_distribution_metadata():
    pyproject = tomllib.loads((Path(__file__).parents[2] / "pyproject.toml").read_text(encoding="utf-8"))
    assert modelport.__version__ == pyproject["project"]["version"]
