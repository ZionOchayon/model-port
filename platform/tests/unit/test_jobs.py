from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from modelport.errors import RemoteOperationError, WaitTimeoutError
from modelport_platform.ray.jobs import JobTimings, SharedClusterJobs


class FakeClient:
    """JobSubmissionClient: jobs by submission ID; `script` maps an ID to the statuses get_job_info walks."""

    def __init__(self, jobs=None, script=None):
        self.jobs = dict(jobs or {})
        self.script = {k: list(v) for k, v in (script or {}).items()}
        self.calls = []

    def get_job_info(self, submission_id):
        self.calls.append(("info", submission_id))
        if self.script.get(submission_id):
            self.jobs[submission_id] = self.script[submission_id].pop(0)
        if submission_id not in self.jobs:
            raise RuntimeError(f"Job {submission_id} does not exist")  # Ray raises on unknown IDs
        status, start, end = self.jobs[submission_id]
        return SimpleNamespace(status=SimpleNamespace(value=status), start_time=start, end_time=end)

    def submit_job(self, **kwargs):
        self.calls.append(("submit", kwargs["submission_id"]))
        self.jobs[kwargs["submission_id"]] = ("PENDING", None, None)
        self.submitted = kwargs
        return kwargs["submission_id"]

    def delete_job(self, submission_id):
        self.calls.append(("delete", submission_id))
        self.jobs.pop(submission_id)
        return True

    def stop_job(self, submission_id):
        self.calls.append(("stop", submission_id))
        return True

    def get_job_logs(self, submission_id):
        return "\n".join(f"line {i}" for i in range(80))


def jobs(client, clock=None):
    return SharedClusterJobs("http://shared:8265", client=client, sleep=lambda s: None, clock=clock)


ARGS = {
    "entrypoint": "python -m modelport_platform.gate.evaluate",
    "runtime_env": {"pip": {"packages": ["xgboost==3.4.1"]}},
    "entrypoint_num_cpus": 1.0,
    "metadata": {"model": "m"},
}


def kinds(client):
    return [c[0] for c in client.calls if c[0] != "info"]


def test_submit_new():
    client = FakeClient()
    assert jobs(client).submit_idempotent(submission_id="eval-m-1", **ARGS) == "eval-m-1"
    assert kinds(client) == ["submit"]
    assert client.submitted["entrypoint_num_cpus"] == 1.0
    assert client.submitted["runtime_env"]["pip"] == {"packages": ["xgboost==3.4.1"]}


def test_skip_succeeded():
    client = FakeClient({"eval-m-1": ("SUCCEEDED", 1000, 2000)})
    jobs(client).submit_idempotent(submission_id="eval-m-1", **ARGS)
    assert kinds(client) == []


def test_force_resubmits_succeeded_but_adopts_running():
    client = FakeClient({"a": ("SUCCEEDED", 1, 2), "b": ("RUNNING", 1, None)})
    jobs(client).submit_idempotent(submission_id="a", force=True, **ARGS)
    jobs(client).submit_idempotent(submission_id="b", force=True, **ARGS)
    assert kinds(client) == ["delete", "submit"]


def test_adopt_running():
    client = FakeClient({"eval-m-1": ("RUNNING", 1000, None)})
    jobs(client).submit_idempotent(submission_id="eval-m-1", **ARGS)
    assert kinds(client) == []


def test_resubmit_failed_deletes_first():
    client = FakeClient({"eval-m-1": ("FAILED", 1000, 2000)})
    jobs(client).submit_idempotent(submission_id="eval-m-1", **ARGS)
    assert kinds(client) == ["delete", "submit"]


def test_wait_failed_raises_with_logs():
    client = FakeClient(script={"j": [("RUNNING", 1000, None), ("FAILED", 1000, 3000)]})
    with pytest.raises(RemoteOperationError) as caught:
        jobs(client).wait("j", timeout_s=60)
    assert "line 79" in caught.value.what
    assert "line 29" not in caught.value.what  # only the last 50 lines


def test_wait_timeout_stops_job():
    ticks = iter(range(0, 1000, 10))
    client = FakeClient({"j": ("RUNNING", 1000, None)})
    with pytest.raises(WaitTimeoutError):
        jobs(client, clock=lambda: next(ticks)).wait("j", timeout_s=30)
    assert ("stop", "j") in client.calls


def test_timings_from_job_details():
    client = FakeClient(
        script={"j": [("PENDING", None, None), ("SUCCEEDED", 1_764_237_610_000, 1_764_237_670_500)]}
    )
    timings = jobs(client).wait("j", timeout_s=60)
    assert timings.started_at == datetime.fromtimestamp(1_764_237_610, UTC)
    assert timings.duration_s == pytest.approx(60.5)
    submitted = datetime.fromtimestamp(1_764_237_600, UTC)
    t = JobTimings(submitted_at=submitted, started_at=timings.started_at, ended_at=timings.ended_at)
    assert t.startup_s == pytest.approx(10.0)
    assert JobTimings(submitted, None, None).duration_s is None
