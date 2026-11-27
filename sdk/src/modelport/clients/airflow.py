"""Airflow REST API v2 client (Adapter behind the `Orchestrator` port)."""

import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal, Protocol

import requests

from modelport.errors import (
    ConfigError,
    NotFoundError,
    PlatformUnavailableError,
    RemoteOperationError,
    WaitTimeoutError,
)
from modelport.logs import get_logger

_log = get_logger(__name__)
RunState = Literal["queued", "running", "success", "failed"]


@dataclass(frozen=True)
class DagRunRef:
    """A triggered (or already existing) DAG run."""

    dag_id: str
    run_id: str
    existed: bool
    url: str


@dataclass(frozen=True)
class DagRunInfo:
    """One DAG run's state."""

    dag_id: str
    run_id: str
    state: RunState
    start: datetime | None
    end: datetime | None
    conf: Mapping[str, Any]
    url: str


@dataclass(frozen=True)
class TaskState:
    """One task instance of a DAG run."""

    task_id: str
    state: str | None
    try_number: int


class Orchestrator(Protocol):
    """What the CLI needs from the workflow engine."""

    def trigger(self, dag_id: str, run_id: str, conf: Mapping[str, Any]) -> DagRunRef:
        """Start a run with this ID, or return the existing one."""

    def has_dag(self, dag_id: str) -> bool:
        """Whether the DAG is installed."""

    def get_run(self, dag_id: str, run_id: str) -> DagRunInfo | None:
        """The run, or None when it does not exist (yet)."""

    def xcom(self, dag_id: str, run_id: str, task_id: str, key: str = "return_value") -> Any:  # noqa: ANN401
        """A task's XCom value as JSON, or None when the task pushed none."""

    def task_states(self, dag_id: str, run_id: str) -> list[TaskState]:
        """Every task instance of the run."""

    def wait(self, dag_id: str, run_id: str, *, timeout_s: float, poll_s: float = 10.0) -> DagRunInfo:
        """Block until the run succeeds; raise when it fails or the timeout passes."""


def _time(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value.replace("Z", "+00:00")) if value else None


class AirflowClient:
    """Token auth (`POST /auth/token`), refreshed once on a 401."""

    def __init__(
        self,
        base_url: str,
        username: str,
        password: str,
        *,
        timeout_s: float = 30.0,
        session: requests.Session | None = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self._credentials = {"username": username, "password": password}
        self._timeout_s = timeout_s
        self._session = session or requests.Session()
        self._sleep, self._clock = sleep, clock
        self._token: str | None = None

    def _send(
        self, method: str, path: str, *, json: object = None, headers: Mapping[str, str] | None = None
    ) -> Any:  # noqa: ANN401
        url = f"{self.base_url}{path}"
        try:
            return self._session.request(
                method, url, json=json, headers=dict(headers or {}), timeout=self._timeout_s
            )
        except requests.RequestException as err:
            raise PlatformUnavailableError(
                what=f"Cannot reach Airflow at {self.base_url} ({type(err).__name__})",
                fix="Run `make ui` to open the port-forwards (after `make up`).",
            ) from err

    def _fetch_token(self) -> str:
        response = self._send("POST", "/auth/token", json=self._credentials)
        if response.status_code in (401, 403):
            raise ConfigError(
                what=f"Airflow at {self.base_url} rejected user '{self._credentials['username']}'",
                fix="Use the credentials in .modelport/credentials.env (written by `make up`).",
            )
        if response.status_code >= 400:
            raise RemoteOperationError(
                what=f"Airflow token request failed with HTTP {response.status_code}",
                fix="Check that Airflow is up (`make ui`) and retry; --debug shows the HTTP details.",
            )
        return str(response.json()["access_token"])

    def _request(self, method: str, path: str, json: Any = None) -> Any:  # noqa: ANN401
        for attempt in range(2):
            if self._token is None:
                self._token = self._fetch_token()
            response = self._send(method, path, json=json, headers={"Authorization": f"Bearer {self._token}"})
            if response.status_code == 401 and attempt == 0:
                self._token = None  # expired: fetch a new token once
                continue
            if response.status_code in (401, 403):
                raise ConfigError(
                    what=f"Airflow refused {method} {path} (HTTP {response.status_code})",
                    fix="Use the credentials in .modelport/credentials.env (written by `make up`).",
                )
            return response
        raise AssertionError("unreachable")  # pragma: no cover

    def _url(self, dag_id: str, run_id: str) -> str:
        return f"{self.base_url}/dags/{dag_id}/runs/{run_id}"

    @staticmethod
    def _check(response: Any, what: str) -> None:  # noqa: ANN401
        if response.status_code >= 400:
            raise RemoteOperationError(
                what=f"{what} failed with HTTP {response.status_code}: {response.text[:300]}",
                fix="Check that Airflow is up (`make ui`) and retry; --debug shows the HTTP details.",
            )

    def trigger(self, dag_id: str, run_id: str, conf: Mapping[str, Any]) -> DagRunRef:
        """Start the run; a run with the same ID already existing is not an error (idempotent submit)."""
        body = {"dag_run_id": run_id, "logical_date": None, "conf": dict(conf)}
        response = self._request("POST", f"/api/v2/dags/{dag_id}/dagRuns", body)
        if response.status_code == 404:
            raise NotFoundError(
                what=f"Airflow has no DAG '{dag_id}'", fix="Run `make up`; DAGs ship in the Airflow image."
            )
        existed = response.status_code == 409
        if not existed:
            self._check(response, f"Triggering {dag_id}")
        _log.info("airflow.triggered", dag_id=dag_id, run_id=run_id, existed=existed)
        return DagRunRef(dag_id, run_id, existed, self._url(dag_id, run_id))

    def has_dag(self, dag_id: str) -> bool:
        """Whether Airflow knows the DAG."""
        response = self._request("GET", f"/api/v2/dags/{dag_id}")
        if response.status_code == 404:
            return False
        self._check(response, f"Reading DAG {dag_id}")
        return True

    def get_run(self, dag_id: str, run_id: str) -> DagRunInfo | None:
        """The run, or None."""
        response = self._request("GET", f"/api/v2/dags/{dag_id}/dagRuns/{run_id}")
        if response.status_code == 404:
            return None
        self._check(response, f"Reading run {run_id}")
        body = response.json()
        return DagRunInfo(
            dag_id=dag_id,
            run_id=run_id,
            state=body["state"],
            start=_time(body.get("start_date")),
            end=_time(body.get("end_date")),
            conf=body.get("conf") or {},
            url=self._url(dag_id, run_id),
        )

    def xcom(self, dag_id: str, run_id: str, task_id: str, key: str = "return_value") -> Any:  # noqa: ANN401
        """A task's XCom value as JSON, or None when the task pushed none."""
        path = f"/api/v2/dags/{dag_id}/dagRuns/{run_id}/taskInstances/{task_id}/xcomEntries/{key}"
        response = self._request("GET", f"{path}?deserialize=true&stringify=false")
        if response.status_code == 404:
            return None
        self._check(response, f"Reading XCom {task_id} of {run_id}")
        return response.json().get("value")

    def task_states(self, dag_id: str, run_id: str) -> list[TaskState]:
        """Every task instance of the run."""
        response = self._request("GET", f"/api/v2/dags/{dag_id}/dagRuns/{run_id}/taskInstances")
        self._check(response, f"Reading tasks of {run_id}")
        return [
            TaskState(t["task_id"], t.get("state"), int(t.get("try_number") or 0))
            for t in response.json().get("task_instances", [])
        ]

    def wait(self, dag_id: str, run_id: str, *, timeout_s: float, poll_s: float = 10.0) -> DagRunInfo:
        """Poll until the run succeeds (a run not created yet counts as not finished).

        Raises:
            RemoteOperationError: the run failed (names the first failed task).
            WaitTimeoutError: `timeout_s` passed first.
        """
        deadline = self._clock() + timeout_s
        while True:
            info = self.get_run(dag_id, run_id)
            if info is not None and info.state == "success":
                return info
            if info is not None and info.state == "failed":
                failed = next(
                    (t.task_id for t in self.task_states(dag_id, run_id) if t.state == "failed"), "?"
                )
                raise RemoteOperationError(
                    what=f"Airflow run {run_id} failed at task {failed}",
                    fix=f"Open {info.url} to see the log; `modelport status <model>` shows the reason.",
                )
            if self._clock() >= deadline:
                raise WaitTimeoutError(
                    what=f"Airflow run {run_id} did not finish within {timeout_s:.0f} s",
                    fix=f"It keeps running: open {self._url(dag_id, run_id)} or `modelport status <model>`.",
                )
            self._sleep(poll_s)
