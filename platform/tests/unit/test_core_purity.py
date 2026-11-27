"""Pure core modules import no I/O library (hexagonal core), checked in a clean subprocess."""

import importlib.util
import subprocess
import sys

import pytest

BLOCKED = ("mlflow", "boto3", "botocore", "kubernetes", "ray", "airflow", "requests", "pyarrow.fs")
CORE = (
    "modelport.lifecycle",
    "modelport_platform.ingest.quality",
    "modelport_platform.gate.core",
    "modelport_platform.gate.metrics",
    "modelport_platform.canary.core",
    "modelport_platform.drift.psi",
    "modelport_platform.serve.config",
    "modelport_platform.ray.manifests",
)
PRESENT = [m for m in CORE if importlib.util.find_spec(m.rsplit(".", 1)[0]) and importlib.util.find_spec(m)]


def test_core_list_is_checked():
    assert "modelport.lifecycle" in PRESENT


@pytest.mark.parametrize("module", PRESENT)
def test_core_modules_import_without_io_libs(module):
    code = (
        "import sys\n" + "".join(f"sys.modules[{name!r}] = None\n" for name in BLOCKED) + f"import {module}\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr[-2000:]
