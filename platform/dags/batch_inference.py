"""batch_inference: score a day's input for every champion on the shared RayCluster (FR-12).

`plan` locks each champion's version, so a promotion while the run is in progress does not change what
the run scores. Each job overwrites its output folder and then writes `_SUCCESS.json`: rerunning a day
replaces its predictions and never duplicates them.
"""

import json
from collections.abc import Mapping, Sequence
from datetime import UTC, date, datetime, timedelta
from typing import Any

from airflow.sdk import Param, dag, task

from modelport.logs import get_logger
from modelport.naming import Tag, batch_input_uri, prediction_uri, shared_job_id
from modelport_platform import dagkit
from modelport_platform.dagkit import Pool, ServeRegistry
from modelport_platform.metrics.push import push_job_timing
from modelport_platform.ports import DatasetStore

_log = get_logger(__name__)
BATCH_TIMEOUT_S = 3600


def plan(
    reg: ServeRegistry, store: DatasetStore, models: Sequence[str] | None, day: date
) -> list[dict[str, Any]]:
    """One job per champion with input for `day`, each pinned to the champion version found now."""
    planned = []
    for model in models or reg.list_models():
        champion = reg.snapshot(model).champion
        if champion is None:
            continue
        info = reg.get_version(model, champion)
        spec = dagkit.spec_for_version(reg, info)
        input_uri = batch_input_uri(spec.data.dataset_name, day)
        if not store.list(input_uri):
            _log.info("batch.no_input", model=model, input_uri=input_uri)
            continue
        planned.append(
            {
                "model": model,
                "version": champion,
                "day": day.isoformat(),
                "input_uri": input_uri,
                "output_uri": prediction_uri(model, day),
                "runtime_pip": list(json.loads(info.tags.get(Tag.RUNTIME_PIP) or "[]")),
                "batch_size": spec.resources.batch.batch_size,
                "num_cpus": spec.resources.batch.cpus_per_task,
            }
        )
    return planned


def batch_job(item: Mapping[str, Any], *, force: bool) -> dict[str, Any]:
    """The `submit_idempotent` arguments of one planned job (ID by model, day and locked version)."""
    settings = dagkit.settings()
    runtime_env: dict[str, Any] = {
        # non-secret settings only: credentials come from the pods' Secret
        "env_vars": {
            "MLFLOW_TRACKING_URI": settings.mlflow_uri,
            "MODELPORT_S3_ENDPOINT": settings.s3_endpoint,
            "AWS_ENDPOINT_URL": settings.s3_endpoint,
        },
    }
    if item["runtime_pip"]:
        runtime_env["pip"] = {"packages": list(item["runtime_pip"]), "pip_check": False}
    return {
        "submission_id": shared_job_id("batch", item["model"], f"{item['day']}-v{item['version']}"),
        "entrypoint": (
            f"python -m modelport_platform.batch.job --model {item['model']} --version {item['version']}"
            f" --input-uri {item['input_uri']} --output-uri {item['output_uri']} --day {item['day']}"
            f" --batch-size {item['batch_size']} --num-cpus {item['num_cpus']}"
        ),
        "runtime_env": runtime_env,
        # the driver only drives Ray Data; a reserved CPU starved its ListFiles task on the 2-CPU smoke worker
        "entrypoint_num_cpus": 0,
        "metadata": {"model": item["model"], "version": item["version"], "kind": "batch"},
        "force": force,
    }


def run_day(day_param: str | None, logical_date: datetime | None, run_after: datetime | None) -> date:
    """`day` param, else the run's logical date, else (a manual run without one) the day it ran."""
    if day_param:
        return date.fromisoformat(day_param)
    return (logical_date or run_after or datetime.now(UTC)).date()


@dag(
    dag_id="batch_inference",
    schedule="@daily",
    start_date=datetime(2026, 1, 1, tzinfo=UTC),
    max_active_runs=1,
    catchup=False,
    default_args=dagkit.DEFAULT_ARGS,
    tags=[*dagkit.TAGS, "batch"],
    doc_md=__doc__,
    params={
        "models": Param(None, type=["null", "array"], items={"type": "string"}),
        "day": Param(None, type=["null", "string"], pattern=r"^\d{4}-\d{2}-\d{2}$"),
        "force_rerun": Param(False, type="boolean"),  # re-execute jobs that already succeeded
    },
)
def batch_inference() -> None:
    """Plan (versions locked) -> one shared-cluster job per model -> summary."""

    @task(task_id="plan")
    def plan_task(params: Any = None, logical_date: Any = None, dag_run: Any = None) -> list[dict[str, Any]]:  # noqa: ANN401
        day = run_day(params["day"], logical_date, getattr(dag_run, "run_after", None))
        return plan(dagkit.registry(), dagkit.store(), params["models"], day)

    @task(task_id="run_batch", pool=Pool.SHARED_CLUSTER, execution_timeout=timedelta(seconds=BATCH_TIMEOUT_S))
    def run_task(item: dict[str, Any], params: Any = None) -> dict[str, Any]:  # noqa: ANN401
        jobs, settings = dagkit.jobs(), dagkit.settings()
        job = batch_job(item, force=bool(params["force_rerun"]))
        jobs.submit_idempotent(**job)
        timings = jobs.wait(job["submission_id"], timeout_s=BATCH_TIMEOUT_S)
        push_job_timing(
            settings.pushgateway_url,
            model=item["model"],
            pattern="shared",
            kind="batch",
            startup_s=timings.startup_s,
            duration_s=timings.duration_s,
            cpu_s=None,
        )
        marker = json.loads(dagkit.store().read_bytes(f"{item['output_uri']}_SUCCESS.json"))
        return {"model": item["model"], "version": item["version"], "rows": marker["rows"]}

    @task(task_id="summary")
    def summary_task(results: list[dict[str, Any]]) -> None:
        for result in results:
            _log.info("batch.summary", **result)

    summary_task(run_task.expand(item=plan_task()))


DAG = batch_inference()
