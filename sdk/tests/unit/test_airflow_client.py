import pytest
import requests

from modelport.clients.airflow import AirflowClient
from modelport.errors import (
    ConfigError,
    NotFoundError,
    PlatformUnavailableError,
    RemoteOperationError,
    WaitTimeoutError,
)

BASE = "http://airflow:8080"


class Response:
    def __init__(self, status, payload=None):
        self.status_code, self._payload = status, payload or {}
        self.text = str(payload)

    def json(self):
        return self._payload


class Session:
    """Answers (method, path) from a script; each entry is a Response or a list consumed in order."""

    def __init__(self, routes):
        self.routes, self.calls = routes, []

    def request(self, method, url, *, json=None, headers=None, timeout=None):
        path = url.removeprefix(BASE)
        self.calls.append((method, path, (headers or {}).get("Authorization"), json))
        answer = self.routes[(method, path)]
        if isinstance(answer, Exception):
            raise answer
        return answer.pop(0) if isinstance(answer, list) else answer


TOKEN = ("POST", "/auth/token")
RUN = "/api/v2/dags/train_model/dagRuns/submit__m__abc"


def client(routes, **kw):
    session = Session({TOKEN: Response(201, {"access_token": "t1"}), **routes})
    return AirflowClient(BASE, "modelport", "pw", session=session, sleep=lambda s: None, **kw), session


def test_token_fetched_and_cached():
    c, s = client(
        {("GET", RUN): Response(200, {"dag_run_id": "submit__m__abc", "state": "running", "conf": {}})}
    )
    c.get_run("train_model", "submit__m__abc")
    c.get_run("train_model", "submit__m__abc")
    assert [call[:2] for call in s.calls].count(TOKEN) == 1
    assert s.calls[0][3] == {"username": "modelport", "password": "pw"}
    assert s.calls[1][2] == "Bearer t1"


def test_token_refreshed_on_401():
    c, s = client(
        {("GET", RUN): [Response(401), Response(200, {"dag_run_id": "x", "state": "queued", "conf": {}})]}
    )
    s.routes[TOKEN] = [Response(201, {"access_token": "t1"}), Response(201, {"access_token": "t2"})]
    assert c.get_run("train_model", "submit__m__abc").state == "queued"
    assert s.calls[-1][2] == "Bearer t2"


def test_credentials_rejected_config_error():
    c, s = client({})
    s.routes[TOKEN] = Response(401)
    with pytest.raises(ConfigError, match="Airflow"):
        c.get_run("train_model", "submit__m__abc")


def test_trigger_409_returns_existed():
    c, s = client({("POST", "/api/v2/dags/train_model/dagRuns"): [Response(200, {}), Response(409, {})]})
    first = c.trigger("train_model", "submit__m__abc", {"model": "m"})
    again = c.trigger("train_model", "submit__m__abc", {"model": "m"})
    assert (first.existed, again.existed) == (False, True)
    assert first.url == f"{BASE}/dags/train_model/runs/submit__m__abc"
    assert s.calls[1][3] == {"dag_run_id": "submit__m__abc", "logical_date": None, "conf": {"model": "m"}}


def test_trigger_404_not_found_error():
    c, _ = client({("POST", "/api/v2/dags/train_model/dagRuns"): Response(404, {})})
    with pytest.raises(NotFoundError, match="Airflow has no DAG 'train_model'"):
        c.trigger("train_model", "submit__m__abc", {})


def test_get_run_missing_is_none_and_has_dag():
    c, _ = client(
        {
            ("GET", RUN): Response(404),
            ("GET", "/api/v2/dags/train_model"): Response(200, {}),
            ("GET", "/api/v2/dags/evaluate_and_promote"): Response(404),
        }
    )
    assert c.get_run("train_model", "submit__m__abc") is None
    assert c.has_dag("train_model") is True
    assert c.has_dag("evaluate_and_promote") is False


def run_state(state):
    return Response(200, {"dag_run_id": "submit__m__abc", "state": state, "conf": {}, "start_date": None})


def test_wait_failed_names_task():
    tasks = {
        "task_instances": [
            {"task_id": "prepare", "state": "success", "try_number": 1},
            {"task_id": "wait_rayjob", "state": "failed", "try_number": 1},
        ]
    }
    c, _ = client(
        {
            ("GET", RUN): [Response(404), run_state("running"), run_state("failed")],
            ("GET", RUN + "/taskInstances"): Response(200, tasks),
        }
    )
    with pytest.raises(RemoteOperationError, match="failed at task wait_rayjob"):
        c.wait("train_model", "submit__m__abc", timeout_s=60, poll_s=1)


def test_wait_success_returns_info():
    c, _ = client({("GET", RUN): [run_state("queued"), run_state("success")]})
    assert c.wait("train_model", "submit__m__abc", timeout_s=60, poll_s=1).state == "success"


def test_wait_timeout():
    clock = iter(range(0, 1000, 10))
    c, _ = client({("GET", RUN): run_state("running")}, clock=lambda: next(clock))
    with pytest.raises(WaitTimeoutError, match="submit__m__abc"):
        c.wait("train_model", "submit__m__abc", timeout_s=30, poll_s=10)


def test_connection_error_platform_unavailable():
    c, s = client({})
    s.routes[TOKEN] = requests.ConnectionError("refused")
    with pytest.raises(PlatformUnavailableError, match="make ui"):
        c.get_run("train_model", "submit__m__abc")


def test_xcom_value_and_missing():
    path = RUN + "/taskInstances/wait_rayjob/xcomEntries/return_value?deserialize=true&stringify=false"
    c, _ = client({("GET", path): [Response(200, {"value": {"job_status": "SUCCEEDED"}}), Response(404)]})
    assert c.xcom("train_model", "submit__m__abc", "wait_rayjob") == {"job_status": "SUCCEEDED"}
    assert c.xcom("train_model", "submit__m__abc", "wait_rayjob") is None
