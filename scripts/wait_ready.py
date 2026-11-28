"""Wait until the modelport stack is ready (step 14 of `make up`).

Usage: uv run python scripts/wait_ready.py --profile smoke|full --timeout SECONDS
Exit 0 when ready, 124 on timeout. Prints what is not ready every 30 s.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
NAMESPACES = ("platform", "ray", "monitoring")
RAY_CRDS = ("rayjobs.ray.io", "rayservices.ray.io", "rayclusters.ray.io")
PROMETHEUS_CRD = "prometheusrules.monitoring.coreos.com"
KUBECTL = ("kubectl", "--context", "kind-modelport")


def not_ready_pods(pods: Mapping[str, Any]) -> list[str]:
    """Names of pods that are not Ready, ignoring pods of finished Jobs (phase Succeeded)."""
    problems = []
    for pod in pods.get("items", []):
        if pod.get("status", {}).get("phase") == "Succeeded":
            continue
        conditions = {c["type"]: c["status"] for c in pod.get("status", {}).get("conditions", [])}
        if conditions.get("Ready") != "True":
            meta = pod["metadata"]
            problems.append(f"pod {meta['namespace']}/{meta['name']} not Ready")
    return problems


def dag_problems(
    listed: Sequence[Mapping[str, Any]], import_errors: Sequence[Any], expected: set[str]
) -> list[str]:
    """Missing DAG IDs and import errors reported by the Airflow CLI."""
    missing = sorted(expected - {d["dag_id"] for d in listed})
    problems = [f"DAG {d} not loaded" for d in missing]
    problems += [f"DAG import error: {e}" for e in import_errors]
    return problems


def expected_dags() -> set[str]:
    """DAG IDs are the file stems in platform/dags (each file defines the DAG of the same name)."""
    return {p.stem for p in (ROOT / "platform" / "dags").glob("*.py")}


def run(*args: str) -> subprocess.CompletedProcess[str]:
    """Run kubectl against the modelport cluster and capture output."""
    return subprocess.run([*KUBECTL, *args], capture_output=True, text=True, check=False)  # noqa: S603


def check_once() -> list[str]:
    """Every readiness problem right now; empty when the stack is ready."""
    problems: list[str] = []
    for ns in NAMESPACES:
        out = run("get", "pods", "-n", ns, "-o", "json")
        problems += (
            not_ready_pods(json.loads(out.stdout)) if out.returncode == 0 else [f"cannot list pods in {ns}"]
        )
    crds = [
        *RAY_CRDS,
        *([PROMETHEUS_CRD] if (ROOT / "deploy/monitoring/kustomization.yaml").exists() else []),
    ]
    problems += [f"CRD {c} missing" for c in crds if run("get", "crd", c).returncode != 0]
    if (ROOT / "deploy/manifests/base/shared-raycluster.yaml").exists():
        state = run(
            "get", "raycluster", "-n", "ray", "modelport-shared", "-o", "jsonpath={.status.state}"
        ).stdout
        if state != "ready":
            problems.append(f"RayCluster modelport-shared is '{state or 'absent'}'")
    probe = "import urllib.request; urllib.request.urlopen('http://localhost:5000/health')"
    if run("exec", "-n", "platform", "deploy/mlflow", "--", "python", "-c", probe).returncode != 0:
        problems.append("MLflow /health not answering")
    problems += airflow_problems()
    return problems


def airflow_problems() -> list[str]:
    """DAGs loaded and no import errors, asked through the scheduler pod."""
    base = ("exec", "-n", "platform", "deploy/airflow-scheduler", "-c", "scheduler", "--", "airflow", "dags")
    listed = run(*base, "list", "-o", "json")
    errors = run(*base, "list-import-errors", "-o", "json")
    failed = next((out for out in (listed, errors) if out.returncode != 0), None)
    if failed is not None:
        tail = " ".join(failed.stderr.strip().splitlines()[-2:]) or "no stderr"
        return [f"Airflow CLI not answering in the scheduler pod (exit {failed.returncode}: {tail})"]
    return dag_problems(json.loads(listed.stdout or "[]"), json.loads(errors.stdout or "[]"), expected_dags())


def wait(
    timeout_s: float,
    check: Callable[[], list[str]] = check_once,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> int:
    """Poll `check` every 10 s until it reports nothing (0) or the timeout passes (124)."""
    deadline = clock() + timeout_s
    last_report = float("-inf")
    while True:
        problems = check()
        if not problems:
            return 0
        now = clock()
        if now >= deadline:
            print("wait_ready: timed out; not ready:\n  " + "\n  ".join(problems), file=sys.stderr)
            return 124
        if now - last_report >= 30:
            print("wait_ready: waiting for:\n  " + "\n  ".join(problems))
            last_report = now
        sleep(10)


def main(argv: Sequence[str] | None = None) -> int:
    """Parse arguments and wait."""
    parser = argparse.ArgumentParser(description="Wait until the modelport stack is ready.")
    parser.add_argument("--profile", choices=("smoke", "full"), required=True)
    parser.add_argument("--timeout", type=float, default=900)
    args = parser.parse_args(argv)
    code = wait(args.timeout)
    if code == 0:
        print(f"wait_ready: all components ready (profile {args.profile})")
    return code


if __name__ == "__main__":
    sys.exit(main())
