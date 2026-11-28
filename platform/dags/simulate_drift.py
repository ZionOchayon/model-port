"""simulate_drift: the drift demo; send deliberately drifted requests to a champion, then check drift (FR-16).

Run it by hand. It takes `rows` rows of the champion's validation set, perturbs them (`brightness`, `noise`
or `invert` for an image model, `shift_feature` for a tabular one), sends them to the model's endpoint at
`rps` requests a second for `duration_minutes`, then starts `drift_check` over the last hour. The batch
drift alert fires as soon as that check pushes a critical PSI; the live alert after its 15 minutes.
"""

import io
from datetime import UTC, datetime
from typing import Any

import pandas as pd
from airflow.providers.standard.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.sdk import Param, dag, task
from airflow.sdk.exceptions import AirflowFailException

from modelport.clients.serve import ServeClient
from modelport.logs import get_logger
from modelport.naming import MODEL_NAME_PATTERN, Tag
from modelport_platform import dagkit
from modelport_platform.dagkit import ServeRegistry
from modelport_platform.drift.simulate import Perturbation, check_perturbation, perturb
from modelport_platform.ports import DatasetStore
from modelport_platform.tools.loadgen import run_load

_log = get_logger(__name__)


def load_rows(
    reg: ServeRegistry, store: DatasetStore, model: str, perturbation: str, feature: str | None, *, rows: int
) -> dict[str, Any]:
    """Check the perturbation fits the champion and its validation set exists; fails without retries."""
    champion = reg.snapshot(model).champion
    if champion is None:
        raise AirflowFailException(
            f"{model} has no champion: nothing serves it, so there is nothing to drift"
        )
    info = reg.get_version(model, champion)
    try:
        check_perturbation(dagkit.spec_for_version(reg, info), Perturbation(perturbation), feature)
    except ValueError as err:
        raise AirflowFailException(str(err)) from err
    val_uri = info.tags[Tag.VAL_URI]
    available = len(pd.read_parquet(io.BytesIO(store.read_bytes(val_uri))))
    return {"val_uri": val_uri, "rows": min(rows, available)}


@dag(
    dag_id="simulate_drift",
    schedule=None,
    start_date=datetime(2026, 1, 1, tzinfo=UTC),
    catchup=False,
    default_args=dagkit.DEFAULT_ARGS,
    tags=[*dagkit.TAGS, "drift"],
    doc_md=__doc__,
    params={
        "model": Param(type="string", pattern=MODEL_NAME_PATTERN),
        "perturbation": Param("brightness", type="string", enum=[str(p) for p in Perturbation]),
        "feature": Param(None, type=["null", "string"]),
        "rps": Param(5.0, type="number", exclusiveMinimum=0, maximum=50),
        "duration_minutes": Param(5, type="integer", minimum=1, maximum=30),
        "rows": Param(2000, type="integer", minimum=1),
    },
)
def simulate_drift() -> None:
    """Load and check, send drifted traffic, then trigger drift_check."""

    @task(task_id="load_rows")
    def load_task(params: Any = None) -> dict[str, Any]:  # noqa: ANN401
        return load_rows(
            dagkit.registry(),
            dagkit.store(),
            params["model"],
            params["perturbation"],
            params["feature"],
            rows=int(params["rows"]),
        )

    @task(task_id="send_drifted_traffic")
    def send_task(loaded: dict[str, Any], params: Any = None) -> dict[str, Any]:  # noqa: ANN401
        reg, store, settings = dagkit.registry(), dagkit.store(), dagkit.settings()
        model = params["model"]
        champion = reg.snapshot(model).champion
        assert champion is not None  # load_rows checked it
        spec = dagkit.spec_for_version(reg, reg.get_version(model, champion))
        rows = pd.read_parquet(io.BytesIO(store.read_bytes(loaded["val_uri"]))).head(loaded["rows"])
        drifted = perturb(rows, spec, Perturbation(params["perturbation"]), feature=params["feature"])
        result = run_load(
            ServeClient(settings.serve_url),
            model,
            drifted,
            spec_inputs=spec.signature.inputs,
            rps=float(params["rps"]),
            duration_s=int(params["duration_minutes"]) * 60,
        )
        _log.info("simulate.sent", model=model, requests=result.sent, failed=result.failed)
        return {"sent": result.sent, "failed": result.failed, "status_counts": dict(result.status_counts)}

    trigger = TriggerDagRunOperator(
        task_id="trigger_drift_check",
        trigger_dag_id="drift_check",
        trigger_run_id="simulate__{{ params.model }}__{{ ts_nodash }}",
        conf={"models": ["{{ params.model }}"], "window_hours": 1},
    )
    send_task(load_task()) >> trigger


DAG = simulate_drift()
