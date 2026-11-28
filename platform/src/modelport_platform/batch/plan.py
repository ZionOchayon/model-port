"""Which batch jobs a day needs, and the exact shared-cluster submission of each (DAG and bench share it)."""

import json
from collections.abc import Mapping, Sequence
from datetime import date
from typing import Any, Final

from modelport.logs import get_logger
from modelport.naming import Tag, batch_input_uri, prediction_uri, shared_job_id
from modelport_platform import dagkit
from modelport_platform.dagkit import ServeRegistry
from modelport_platform.ports import DatasetStore

_log = get_logger(__name__)
# the driver only drives Ray Data; a reserved CPU starved its ListFiles task on the 2-CPU smoke worker
ENTRYPOINT_NUM_CPUS: Final = 0


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


def batch_job(
    item: Mapping[str, Any], *, force: bool, mlflow_uri: str, s3_endpoint: str, key: str | None = None
) -> dict[str, Any]:
    """The `submit_idempotent` arguments of one planned job (ID by model, day and locked version).

    `mlflow_uri` and `s3_endpoint` are the in-cluster addresses; `key` replaces the ID's day-version part.
    """
    runtime_env: dict[str, Any] = {
        # non-secret settings only: credentials come from the pods' Secret
        "env_vars": {
            "MLFLOW_TRACKING_URI": mlflow_uri,
            "MODELPORT_S3_ENDPOINT": s3_endpoint,
            "AWS_ENDPOINT_URL": s3_endpoint,
        },
    }
    if item["runtime_pip"]:
        runtime_env["pip"] = {"packages": list(item["runtime_pip"]), "pip_check": False}
    return {
        "submission_id": shared_job_id("batch", item["model"], key or f"{item['day']}-v{item['version']}"),
        "entrypoint": (
            f"python -m modelport_platform.batch.job --model {item['model']} --version {item['version']}"
            f" --input-uri {item['input_uri']} --output-uri {item['output_uri']} --day {item['day']}"
            f" --batch-size {item['batch_size']} --num-cpus {item['num_cpus']}"
        ),
        "runtime_env": runtime_env,
        "entrypoint_num_cpus": ENTRYPOINT_NUM_CPUS,
        "metadata": {"model": item["model"], "version": item["version"], "kind": "batch"},
        "force": force,
    }
