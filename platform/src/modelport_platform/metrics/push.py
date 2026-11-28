"""Pushgateway pushes for lifecycle events and job timings.

One group per (job, stable key) and HTTP PUT, so each push replaces its group and no run-specific series
piles up. Without a Pushgateway URL every push is skipped; a failed push is logged, never raised
(metrics are observability, not model state).
"""

import time
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Literal

from prometheus_client import CollectorRegistry, Gauge, delete_from_gateway, push_to_gateway

from modelport.lifecycle import AliasSnapshot
from modelport.logs import get_logger
from modelport.spec import DriftSpec
from modelport_platform.metrics import names

if TYPE_CHECKING:
    from modelport_platform.drift.check import DriftResult

_log = get_logger(__name__)
_SOURCES = ("gate", "canary", "aliases")


def _push(
    url: str | None, job: str, key: Mapping[str, str], fill: Callable[[CollectorRegistry], None]
) -> None:
    if url is None:
        _log.debug("push.skipped", job=job, key=dict(key))
        return
    registry = CollectorRegistry()
    fill(registry)
    try:
        push_to_gateway(url, job=job, registry=registry, grouping_key=dict(key))
    except Exception as err:  # never fail a DAG task over observability
        _log.warning("push.failed", job=job, key=dict(key), error=f"{type(err).__name__}: {err}")


def _result(url: str | None, model: str, source: str, metric: str, stamp: str, passed: bool) -> None:
    def fill(registry: CollectorRegistry) -> None:
        Gauge(metric, "1 passed, 0 failed", registry=registry).set(1.0 if passed else 0.0)
        Gauge(stamp, "Unix time of the result", registry=registry).set(time.time())

    _push(url, names.JOB_LIFECYCLE, {"model": model, "source": source}, fill)


def push_gate_result(url: str | None, model: str, passed: bool) -> None:
    """The latest gate result of a model."""
    _result(url, model, "gate", names.GATE_LAST_RESULT, names.GATE_LAST_RESULT_TIMESTAMP, passed)


def push_canary_result(url: str | None, model: str, passed: bool) -> None:
    """The latest canary result of a model."""
    _result(url, model, "canary", names.CANARY_LAST_RESULT, names.CANARY_LAST_RESULT_TIMESTAMP, passed)


def push_aliases(url: str | None, model: str, snapshot: AliasSnapshot) -> None:
    """The version number each alias points to (an unset alias has no series)."""

    def fill(registry: CollectorRegistry) -> None:
        gauge = Gauge(names.ALIAS_VERSION, "Version an alias points to", ["alias"], registry=registry)
        for alias in ("champion", "previous", "candidate"):
            version = getattr(snapshot, alias)
            if version is not None:
                gauge.labels(alias=alias).set(float(version))

    _push(url, names.JOB_LIFECYCLE, {"model": model, "source": "aliases"}, fill)


def push_job_timing(
    url: str | None,
    *,
    model: str,
    pattern: Literal["rayjob", "shared"],
    kind: Literal["train", "eval", "batch", "bench"],
    startup_s: float | None,
    duration_s: float | None,
    cpu_s: float | None,
) -> None:
    """Startup, duration and CPU seconds of the latest job of this kind (unknown values are left out)."""

    def fill(registry: CollectorRegistry) -> None:
        for metric, value in (
            (names.JOB_STARTUP, startup_s),
            (names.JOB_DURATION, duration_s),
            (names.JOB_CPU, cpu_s),
        ):
            if value is not None:
                Gauge(metric, "Seconds", registry=registry).set(value)

    _push(url, names.JOB_JOBS, {"model": model, "pattern": pattern, "kind": kind}, fill)


def push_drift(url: str | None, model: str, result: "DriftResult", spec: DriftSpec) -> None:
    """PSI per feature, the thresholds and the sample count; a skipped check pushes no PSI.

    The PUT replaces the group, so a skipped check also removes the previous PSI series.
    """

    def fill(registry: CollectorRegistry) -> None:
        psi = Gauge(names.PSI, "PSI against the champion's baseline", ["feature"], registry=registry)
        for feature, value in result.psi.items():
            psi.labels(feature=feature).set(value)
        threshold = Gauge(names.PSI_THRESHOLD, "PSI alert thresholds", ["level"], registry=registry)
        threshold.labels(level="warning").set(spec.psi_warning)
        threshold.labels(level="critical").set(spec.psi_critical)
        Gauge(names.DRIFT_SAMPLES, "Logged rows in the window", registry=registry).set(result.samples)
        if not result.skipped:
            Gauge(names.DRIFT_LAST_SUCCESS, "Unix time of the last PSI", registry=registry).set(time.time())

    _push(url, names.JOB_DRIFT, {"model": model}, fill)


def delete_model_groups(url: str | None, model: str) -> None:
    """Remove a model's lifecycle and drift groups (for a deleted model)."""
    if url is None:
        return
    keys = [(names.JOB_LIFECYCLE, {"model": model, "source": s}) for s in _SOURCES]
    keys.append((names.JOB_DRIFT, {"model": model}))
    for job, key in keys:
        try:
            delete_from_gateway(url, job=job, grouping_key=key)
        except Exception as err:  # same rule as pushes
            _log.warning("push.delete_failed", job=job, key=key, error=f"{type(err).__name__}: {err}")
