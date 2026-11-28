"""drift_check: daily PSI of each champion's live traffic against its training baseline (FR-16).

Reads the inference logs of the last `window_hours` up to `end`, computes PSI per feature for the champion's
version and pushes it to the Pushgateway with the model's thresholds. A model with too few logged rows
pushes its sample count and no PSI. Drift never retrains a model: the alert goes to the data scientist.
"""

from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from typing import Any

from airflow.sdk import Param, dag, task

from modelport.naming import Tag
from modelport_platform import dagkit
from modelport_platform.dagkit import ServeRegistry
from modelport_platform.drift.baseline import Baseline
from modelport_platform.drift.check import compute_drift, load_inference_logs
from modelport_platform.metrics.push import push_drift
from modelport_platform.ports import DatasetStore


def plan(reg: ServeRegistry, models: Sequence[str] | None) -> list[str]:
    """The models (all, or the given ones) that have a champion."""
    return [m for m in models or reg.list_models() if reg.snapshot(m).champion is not None]


def window_end(end_param: str | None, now: datetime) -> datetime:
    """`end` param, else now."""
    return datetime.fromisoformat(end_param) if end_param else now


def check_model(
    reg: ServeRegistry, store: DatasetStore, url: str | None, model: str, end: datetime, *, hours: int
) -> dict[str, Any]:
    """PSI of the champion over the window, pushed; returns a summary for the task log."""
    version = reg.snapshot(model).champion
    assert version is not None  # plan keeps only served models
    info = reg.get_version(model, version)
    spec = dagkit.spec_for_version(reg, info)
    run = info.tags[Tag.TRAIN_RUN_ID]
    baseline = Baseline.from_json(reg.read_json(run, "drift/baseline.json"))
    logs = load_inference_logs(store, model, end - timedelta(hours=hours), end)
    result = compute_drift(logs, baseline, version=version, min_samples=spec.drift.min_samples)
    push_drift(url, model, result, spec.drift)
    return {
        "model": model,
        "version": version,
        "samples": result.samples,
        "skipped": result.skipped,
        "reason": result.reason,
        "max_psi": max(result.psi.values(), default=None),
    }


@dag(
    dag_id="drift_check",
    schedule="@daily",
    start_date=datetime(2026, 1, 1, tzinfo=UTC),
    max_active_runs=1,
    catchup=False,
    default_args=dagkit.DEFAULT_ARGS,
    tags=[*dagkit.TAGS, "drift"],
    doc_md=__doc__,
    params={
        "models": Param(None, type=["null", "array"], items={"type": "string"}),
        "window_hours": Param(24, type="integer", minimum=1, maximum=168),
        "end": Param(None, type=["null", "string"], format="date-time"),
    },
)
def drift_check() -> None:
    """Plan the served models, then one PSI check per model."""

    @task(task_id="plan")
    def plan_task(params: Any = None) -> list[str]:  # noqa: ANN401
        return plan(dagkit.registry(), params["models"])

    @task(task_id="check_model")
    def check_task(model: str, params: Any = None) -> dict[str, Any]:  # noqa: ANN401
        return check_model(
            dagkit.registry(),
            dagkit.store(),
            dagkit.settings().pushgateway_url,
            model,
            window_end(params["end"], datetime.now(UTC)),
            hours=int(params["window_hours"]),
        )

    check_task.expand(model=plan_task())


DAG = drift_check()
