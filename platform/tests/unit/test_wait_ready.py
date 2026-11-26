import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
import wait_ready


def pod(name, phase="Running", ready="True"):
    return {
        "metadata": {"name": name, "namespace": "platform"},
        "status": {"phase": phase, "conditions": [{"type": "Ready", "status": ready}]},
    }


def test_not_ready_pods_ignores_finished_jobs():
    pods = {"items": [pod("ok"), pod("job", phase="Succeeded", ready="False"), pod("bad", ready="False")]}
    assert wait_ready.not_ready_pods(pods) == ["pod platform/bad not Ready"]


def test_dag_problems_lists_missing_and_import_errors():
    problems = wait_ready.dag_problems([{"dag_id": "a"}], [{"filename": "b.py"}], {"a", "b"})
    assert problems == ["DAG b not loaded", "DAG import error: {'filename': 'b.py'}"]


def test_wait_returns_124_on_timeout_and_0_when_ready():
    now = [0.0]

    def sleep(seconds):
        now[0] += seconds

    assert wait_ready.wait(25, check=lambda: ["x"], clock=lambda: now[0], sleep=sleep) == 124
    results = iter([["x"], []])
    assert wait_ready.wait(100, check=lambda: next(results), clock=lambda: now[0], sleep=sleep) == 0
