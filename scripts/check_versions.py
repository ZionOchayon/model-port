"""Check that every pin in the repo agrees with versions.env (NFR-2).

Usage: uv run python scripts/check_versions.py [--kps-chart DIR]

A rule whose file does not exist yet is reported as SKIP; it is enforced as soon as the file is added.
Exit 0 when every present rule holds, 1 otherwise.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAY_LOCK = "deploy/images/ray/requirements.lock"
AIRFLOW_LOCK = "deploy/images/airflow/requirements.lock"

# (pin key, files glob, regex with one group capturing the pinned version). Every capture must equal the pin.
FILE_PINS: tuple[tuple[str, str, str], ...] = (
    ("RAY_VERSION", RAY_LOCK, r"^ray(?:\[[a-z,]+\])?==(\S+)"),
    ("TORCH_VERSION", RAY_LOCK, r"^torch==([\d.]+)"),
    ("TORCHVISION_VERSION", RAY_LOCK, r"^torchvision==([\d.]+)"),
    ("XGBOOST_VERSION", RAY_LOCK, r"^xgboost==(\S+)"),
    ("AIRFLOW_VERSION", AIRFLOW_LOCK, r"^apache-airflow==(\S+)"),
    ("RAY_VERSION", AIRFLOW_LOCK, r"^ray(?:\[[a-z,]+\])?==(\S+)"),
    ("AIRFLOW_VERSION", "deploy/helm/airflow.values.yaml", r'^airflowVersion:\s*"?([^"\s]+)'),
    (
        "RAY_VERSION",
        "platform/src/modelport_platform/ray/manifests.py",
        r'rayVersion["\']?\s*[:=]\s*"([^"]+)"',
    ),
    ("RAY_VERSION", "deploy/manifests/base/shared-raycluster.yaml", r'rayVersion:\s*"([^"]+)"'),
    ("RAY_VERSION", "platform/pyproject.toml", r'"ray\[[a-z,]+\]==([^"]+)"'),
    ("KUBERAY_VERSION", "deploy/helm/kuberay-operator.values.yaml", r"^\s+tag:\s*v(\S+)"),
    ("TORCH_VERSION", "examples/models/*/model*.yaml", r"\btorch==([\w.]+)"),
    ("TORCHVISION_VERSION", "examples/models/*/model*.yaml", r"\btorchvision==([\w.]+)"),
    ("XGBOOST_VERSION", "examples/models/*/model*.yaml", r"\bxgboost==([\w.]+)"),
    ("MLFLOW_VERSION", "pyproject.toml", r"\bmlflow(?:-skinny)?==([\w.]+)"),
    ("MLFLOW_VERSION", "*/pyproject.toml", r"\bmlflow(?:-skinny)?==([\w.]+)"),
    ("MLFLOW_VERSION", "deploy/images/*/requirements.*", r"\bmlflow(?:-skinny)?==([\w.]+)"),
    ("UV_VERSION", ".github/workflows/*.yml", r'astral-sh/setup-uv@\S+\s+with:\s+version:\s*"([^"]+)"'),
    ("TOOLBOX_BASE_IMAGE", "docker-compose.yml", r"^\s+TOOLBOX_BASE_IMAGE:\s*(\S+)"),
    ("DOCKER_CLI_IMAGE", "docker-compose.yml", r"^\s+DOCKER_CLI_IMAGE:\s*(\S+)"),
    ("PROMETHEUS_IMAGE", "docker-compose.yml", r"^\s+PROMETHEUS_IMAGE:\s*(\S+)"),
    ("ALERTMANAGER_IMAGE", "docker-compose.yml", r"^\s+ALERTMANAGER_IMAGE:\s*(\S+)"),
)

# (version key, image key): the image tag must carry the version.
IMAGE_TAGS: tuple[tuple[str, str], ...] = (
    ("RAY_VERSION", "RAY_BASE_IMAGE"),
    ("AIRFLOW_VERSION", "AIRFLOW_BASE_IMAGE"),
    ("MLFLOW_VERSION", "MLFLOW_IMAGE"),
    ("PROMETHEUS_VERSION", "PROMETHEUS_IMAGE"),
    ("ALERTMANAGER_VERSION", "ALERTMANAGER_IMAGE"),
)

# kube-prometheus-stack defaults that must match our pins (checked with --kps-chart after `helm pull`).
KPS_DEFAULTS: tuple[tuple[str, str], ...] = (
    ("PROMETHEUS_VERSION", "prometheus/prometheus"),
    ("ALERTMANAGER_VERSION", "prometheus/alertmanager"),
)


@dataclass(frozen=True)
class Result:
    """Outcome of one rule."""

    rule: str
    ok: bool
    detail: str = ""
    skipped: bool = False


def read_versions(path: Path) -> dict[str, str]:
    """Parse KEY=VALUE lines of versions.env, ignoring comments and blank lines."""
    pins: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line and not line.startswith("#"):
            key, _, value = line.partition("=")
            pins[key.strip()] = value.strip()
    return pins


def compare(rule: str, found: list[str], expected: str) -> Result:
    """Every captured value must equal the pin, and there must be at least one."""
    wrong = sorted(set(found) - {expected})
    if not found:
        return Result(rule, False, "no pin found")
    return Result(rule, not wrong, f"found {', '.join(wrong)}, versions.env has {expected}" if wrong else "")


def file_pin_results(pins: dict[str, str], key: str, glob: str, pattern: str) -> Iterator[Result]:
    """One result per file matching `glob`, or one SKIP when no file exists yet."""
    paths = sorted(ROOT.glob(glob))
    if not paths:
        yield Result(f"{key} in {glob}", True, "not present yet", skipped=True)
    for path in paths:
        rel = path.relative_to(ROOT).as_posix()
        found = re.findall(pattern, path.read_text(encoding="utf-8"), re.MULTILINE)
        if found or "*" not in glob:  # a glob match without that pin (e.g. a model without torch) is fine
            yield compare(f"{key} in {rel}", found, pins[key])


def tag_matches(version: str, image: str) -> bool:
    """True when the image tag starts with the version (optionally prefixed with v)."""
    tag = image.split("@", 1)[0].rsplit(":", 1)[-1]
    return re.match(rf"v?{re.escape(version.lstrip('v'))}(?![\d.])", tag) is not None


def rules(pins: dict[str, str], kps_chart: Path | None) -> Iterator[Result]:
    """Every consistency rule of design section 9.1."""
    for key, value in pins.items():
        yield Result(f"{key} is not 'latest'", "latest" not in value.lower(), value)
        if key.endswith("_IMAGE"):
            yield Result(f"{key} pinned by digest", "@sha256:" in value, value)
    for version_key, image_key in IMAGE_TAGS:
        yield Result(f"{version_key} in {image_key}", tag_matches(pins[version_key], pins[image_key]))
    for key, glob, pattern in FILE_PINS:
        yield from file_pin_results(pins, key, glob, pattern)
    if kps_chart is not None:
        values = (kps_chart / "values.yaml").read_text(encoding="utf-8")
        for key, repository in KPS_DEFAULTS:
            found = re.findall(rf"repository:\s*{repository}\s*\n\s+tag:\s*(v[\d.]+)", values)
            yield compare(f"kps default {repository}", found, pins[key])


def main(argv: list[str] | None = None) -> int:
    """Run every rule, print one line each, and return the exit code."""
    parser = argparse.ArgumentParser(description="Check that every pin agrees with versions.env.")
    parser.add_argument(
        "--kps-chart", type=Path, help="untarred kube-prometheus-stack chart (after helm pull)"
    )
    args = parser.parse_args(argv)
    results = list(rules(read_versions(ROOT / "versions.env"), args.kps_chart))
    for r in results:
        status = "SKIP" if r.skipped else ("ok  " if r.ok else "FAIL")
        print(f"{status} {r.rule}" + (f": {r.detail}" if r.detail and not (r.ok and not r.skipped) else ""))
    failed = sum(not r.ok for r in results)
    skipped = sum(r.skipped for r in results)
    print(f"check_versions: {len(results) - failed}/{len(results)} ok ({skipped} not present yet)")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
