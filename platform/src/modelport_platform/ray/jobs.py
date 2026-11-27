"""Evaluation and batch jobs on the shared RayCluster, submitted idempotently by a deterministic ID."""

import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Final

from modelport.errors import RemoteOperationError, WaitTimeoutError
from modelport.logs import get_logger

if TYPE_CHECKING:
    from ray.job_submission import JobSubmissionClient

_log = get_logger(__name__)
TERMINAL: Final = frozenset({"SUCCEEDED", "FAILED", "STOPPED"})
LOG_LINES: Final = 50


@dataclass(frozen=True)
class JobTimings:
    """When a job was submitted, started and ended."""

    submitted_at: datetime
    started_at: datetime | None  # JobDetails.start_time
    ended_at: datetime | None  # JobDetails.end_time

    @property
    def startup_s(self) -> float | None:
        """Submission to start."""
        return (self.started_at - self.submitted_at).total_seconds() if self.started_at else None

    @property
    def duration_s(self) -> float | None:
        """Start to end."""
        if self.started_at and self.ended_at:
            return (self.ended_at - self.started_at).total_seconds()
        return None


def _ms(value: int | None) -> datetime | None:
    return datetime.fromtimestamp(value / 1000, UTC) if value else None


class SharedClusterJobs:
    """Ray Jobs on `modelport-shared` (address: the head's dashboard URL)."""

    def __init__(
        self,
        address: str,
        client: "JobSubmissionClient | None" = None,
        *,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] | None = None,
    ) -> None:
        if client is None:
            from ray.job_submission import JobSubmissionClient

            client = JobSubmissionClient(address)
        self.address, self._client = address, client
        self._sleep, self._clock = sleep, clock or time.monotonic

    def _status(self, submission_id: str) -> tuple[str, Any] | None:
        try:
            info = self._client.get_job_info(submission_id)
        except Exception as err:  # Ray raises a plain exception for an unknown ID
            if "does not exist" in str(err):
                return None
            raise
        return str(info.status.value), info

    def status(self, submission_id: str) -> str:
        """PENDING | RUNNING | SUCCEEDED | FAILED | STOPPED (raises when the job does not exist)."""
        found = self._status(submission_id)
        if found is None:
            raise RemoteOperationError(
                what=f"Shared-cluster job {submission_id} does not exist",
                fix="Rerun the Airflow task that submits it.",
            )
        return found[0]

    def logs_tail(self, submission_id: str, lines: int = LOG_LINES) -> str:
        """The last `lines` lines of the job's driver log."""
        return "\n".join(str(self._client.get_job_logs(submission_id)).splitlines()[-lines:])

    def submit_idempotent(
        self,
        *,
        submission_id: str,
        entrypoint: str,
        runtime_env: Mapping[str, Any],
        entrypoint_num_cpus: float,
        metadata: Mapping[str, str],
        force: bool = False,
    ) -> str:
        """Submit once per ID: adopt a running job, skip a succeeded one, replace a failed one.

        `force` also replaces a succeeded job (a deliberate rerun); a running job is still adopted.
        """
        found = self._status(submission_id)
        if found is not None:
            status = found[0]
            if status not in TERMINAL:
                _log.info("shared_job.adopted", submission_id=submission_id, status=status)
                return submission_id
            if status == "SUCCEEDED" and not force:
                _log.info("shared_job.skipped", submission_id=submission_id, status=status)
                return submission_id
            if status != "SUCCEEDED":
                _log.warning(
                    "shared_job.previous_attempt",
                    submission_id=submission_id,
                    logs=self.logs_tail(submission_id),
                )
            self._client.delete_job(submission_id)
            _log.info("shared_job.resubmitted", submission_id=submission_id, previous=status)
        self._client.submit_job(
            entrypoint=entrypoint,
            runtime_env=dict(runtime_env),
            submission_id=submission_id,
            entrypoint_num_cpus=entrypoint_num_cpus,
            metadata=dict(metadata),
        )
        _log.info("shared_job.submitted", submission_id=submission_id)
        return submission_id

    def wait(
        self,
        submission_id: str,
        *,
        timeout_s: float,
        poll_s: float = 5.0,
        submitted_at: datetime | None = None,
    ) -> JobTimings:
        """Poll until the job succeeds.

        Raises:
            RemoteOperationError: the job failed or was stopped (with its last 50 log lines).
            WaitTimeoutError: `timeout_s` passed first (the job is stopped).
        """
        submitted_at = submitted_at or datetime.now(UTC)
        deadline = self._clock() + timeout_s
        while True:
            found = self._status(submission_id)
            status, info = found if found else ("PENDING", None)  # not visible yet right after submit
            if status == "SUCCEEDED" and info is not None:
                timings = JobTimings(submitted_at, _ms(info.start_time), _ms(info.end_time))
                _log.info(
                    "shared_job.finished",
                    submission_id=submission_id,
                    status=status,
                    duration_s=timings.duration_s,
                )
                return timings
            if status in TERMINAL:
                raise RemoteOperationError(
                    what=f"Shared-cluster job {submission_id} {status}:\n{self.logs_tail(submission_id)}",
                    fix=f"Full log: Ray dashboard (make ui, http://localhost:8265), job {submission_id}.",
                )
            if self._clock() >= deadline:
                self._client.stop_job(submission_id)
                raise WaitTimeoutError(
                    what=f"Shared-cluster job {submission_id} did not finish in {timeout_s:.0f} s (stopped)",
                    fix="Check the shared RayCluster has a free worker; rerun the Airflow task.",
                )
            self._sleep(poll_s)
