"""deploy_candidate: serve a candidate next to the champion, judge it, promote or roll back (FR-10, FR-11).

Serve intent lives in registered-model tags; every change renders the whole RayService from the registry
(pool `serve_config`, one writer at a time). The promote order keeps every request on a RUNNING app:
traffic moves to the already-running candidate app first, the champion app is updated while it is idle,
and the candidate app is removed only after the champion app runs the new version. A rolled-back canary is
an outcome: the run succeeds and the reasons are stored on the version.
"""

import io
import json
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from typing import Any

import pandas as pd
from airflow.sdk import Param, dag, task
from airflow.sdk.exceptions import AirflowSkipException

from modelport.clients.serve import ServeClient
from modelport.lifecycle import Event, ModelState, plan_canary_failed, plan_promote
from modelport.logs import get_logger
from modelport.naming import MODEL_NAME_PATTERN, Tag
from modelport.spec import ModelSpec
from modelport_platform import dagkit
from modelport_platform.canary.core import CanaryPolicy, CanaryStats, judge_canary
from modelport_platform.canary.prometheus import PrometheusClient, canary_stats
from modelport_platform.dagkit import Pool, ServeRegistry
from modelport_platform.metrics.push import push_aliases, push_canary_result
from modelport_platform.ports import DatasetStore, KubeApi, MetricsQuery
from modelport_platform.settings import PlatformSettings
from modelport_platform.tools.loadgen import run_load

_log = get_logger(__name__)
CANARY_RPS = 5.0


def apps_of(model: str) -> dict[str, str]:
    """The Serve application names of one model."""
    return {"router": model, "champion": f"{model}--champion", "candidate": f"{model}--candidate"}


def check_candidate(reg: ServeRegistry, model: str, version: str) -> dict[str, Any]:
    """The version must hold alias `candidate` in state `candidate`; returns what the later tasks need."""
    snapshot = reg.snapshot(model)
    info = reg.get_version(model, version)
    if snapshot.candidate != version or info.state is not ModelState.CANDIDATE:
        raise AirflowSkipException(
            f"v{version} is not the candidate (alias {snapshot.candidate}, {info.state})"
        )
    spec = dagkit.spec_for_version(reg, info)
    return {
        "has_champion": snapshot.champion is not None,
        "window_minutes": spec.serving.observation_window_minutes,
        "canary_weight": spec.serving.canary_weight,
        "val_uri": info.tags[Tag.VAL_URI],
    }


def start_canary(
    reg: ServeRegistry,
    k: KubeApi,
    s: PlatformSettings,
    model: str,
    version: str,
    checked: Mapping[str, Any],
    *,
    fault_error_rate: float,
    now: datetime,
) -> dict[str, Any]:
    """Serve the candidate with its share of traffic (all of it when there is no champion yet)."""
    weight = float(checked["canary_weight"]) if checked["has_champion"] else 1.0
    reg.set_model_tags(
        model,
        {
            Tag.SERVE_CANDIDATE_VERSION: version,
            Tag.SERVE_CANDIDATE_WEIGHT: str(weight),
            Tag.SERVE_FAULT_ERROR_RATE: str(fault_error_rate),
        },
    )
    apps = dagkit.apply_serve_state(reg, k, s)
    _log.info(
        "canary.started", model=model, version=version, weight=weight, fault_error_rate=fault_error_rate
    )
    return {"apps": apps, "started_at": now.isoformat(), "weight": weight}


def canary_traffic(
    client: ServeClient, store: DatasetStore, spec: ModelSpec, model: str, val_uri: str, *, window_s: float
) -> dict[str, Any]:
    """Synthetic requests from the version's validation rows for the whole observation window."""
    rows = pd.read_parquet(io.BytesIO(store.read_bytes(val_uri)))
    result = run_load(
        client, model, rows, spec_inputs=spec.signature.inputs, rps=CANARY_RPS, duration_s=window_s
    )
    _log.info(
        "canary.traffic", model=model, sent=result.sent, failed=result.failed, variants=dict(result.variants)
    )
    return {"sent": result.sent, "failed": result.failed, "status_counts": dict(result.status_counts)}


def judge(
    reg: ServeRegistry,
    query: MetricsQuery,
    model: str,
    version: str,
    *,
    deploy_failed: str | None,
    has_champion: bool,
    window_s: int,
    now: datetime,
    pushgateway_url: str | None,
) -> dict[str, Any]:
    """Judge the candidate on Prometheus (or fail it when its app never deployed); record the verdict."""
    info = reg.get_version(model, version)
    spec = dagkit.spec_for_version(reg, info)
    apps = apps_of(model)
    if deploy_failed is not None:
        passed, reasons = False, [f"candidate app did not deploy: {deploy_failed}"]
        candidate = CanaryStats(0, 0, None)
    else:
        candidate = canary_stats(query, apps["candidate"], window_s, at=now)
        champion = canary_stats(query, apps["champion"], window_s, at=now) if has_champion else None
        verdict = judge_canary(champion, candidate, CanaryPolicy.from_spec(spec))
        passed, reasons = verdict.passed, list(verdict.reasons)
    reg.set_version_tags(
        model,
        version,
        {Tag.CANARY_RESULT: "passed" if passed else "failed", Tag.CANARY_REASONS: json.dumps(reasons)},
    )
    push_canary_result(pushgateway_url, model, passed)
    _log.info(
        "canary.judged",
        model=model,
        version=version,
        passed=passed,
        reasons=reasons,
        candidate_requests=candidate.requests,
        candidate_error_rate=candidate.error_rate,
        candidate_p95_ms=candidate.p95_ms,
    )
    return {"passed": passed, "reasons": reasons}


def promote_aliases(reg: ServeRegistry, model: str, version: str, pushgateway_url: str | None) -> None:
    """Make the candidate the champion; the old champion becomes previous. A retry after that is a no-op."""
    if reg.snapshot(model).champion == version:
        return
    plan = plan_promote(
        reg.snapshot(model), version, event=Event.CANARY_PASSED, current_state=ModelState.CANDIDATE
    )
    reg.apply_plan(model, plan)
    push_aliases(pushgateway_url, model, reg.snapshot(model))
    _log.info("canary.promoted", model=model, version=version)


def apply_now(reg: ServeRegistry, k: KubeApi, s: PlatformSettings) -> list[str]:
    """Render and apply the current serve state (the champion app follows the champion alias)."""
    return dagkit.apply_serve_state(reg, k, s)


def mark_rejected(reg: ServeRegistry, model: str, version: str, pushgateway_url: str | None) -> None:
    """The version is rejected and loses the candidate alias. A retry after that is a no-op."""
    if reg.get_version(model, version).state is ModelState.REJECTED:
        return
    reg.apply_plan(model, plan_canary_failed(reg.snapshot(model), version))
    push_aliases(pushgateway_url, model, reg.snapshot(model))
    _log.info("canary.rolled_back", model=model, version=version)


def branch_choice(verdict: Mapping[str, Any]) -> str:
    """The first task of the promote path or of the rollback path."""
    return "route_all_to_candidate" if verdict["passed"] else "remove_candidate_app"


def window_seconds(params: Mapping[str, Any], checked: Mapping[str, Any]) -> int:
    """`observation_minutes`, else the spec's `serving.observation_window_minutes`, in seconds."""
    return int(params.get("observation_minutes") or checked["window_minutes"]) * 60


@dag(
    dag_id="deploy_candidate",
    schedule=None,
    max_active_runs=1,  # canaries never overlap
    catchup=False,
    default_args=dagkit.DEFAULT_ARGS,
    tags=[*dagkit.TAGS, "serve"],
    doc_md=__doc__,
    params={
        "model": Param(type="string", pattern=MODEL_NAME_PATTERN),
        "version": Param(type="string", pattern=r"^[0-9]+$"),
        "observation_minutes": Param(None, type=["null", "integer"], minimum=1, maximum=120),
        "synthetic_traffic": Param(None, type=["null", "boolean"]),
        "fault_error_rate": Param(0.0, type="number", minimum=0, maximum=1),  # drill only (FR-10)
    },
)
def deploy_candidate() -> None:
    """Canary, judge, then the promote swap or the rollback."""

    @task(task_id="check_candidate")
    def check_task(params: Any = None) -> dict[str, Any]:  # noqa: ANN401
        return check_candidate(dagkit.registry(), params["model"], params["version"])

    @task(task_id="start_canary", pool=Pool.SERVE_CONFIG)
    def start_task(checked: dict[str, Any], params: Any = None) -> dict[str, Any]:  # noqa: ANN401
        return start_canary(
            dagkit.registry(),
            dagkit.kube(),
            dagkit.settings(),
            params["model"],
            params["version"],
            checked,
            fault_error_rate=float(params["fault_error_rate"]),
            now=datetime.now(UTC),
        )

    def ready(apps: list[str], tolerate: tuple[str, ...] = ()) -> Any:  # noqa: ANN401 - PokeReturnValue
        return dagkit.rayservice_ready_poke(dagkit.kube(), apps, tolerate=tolerate)

    @task.sensor(task_id="wait_canary_ready", mode="reschedule", poke_interval=15, timeout=900)
    def wait_canary_task(started: dict[str, Any], params: Any = None) -> Any:  # noqa: ANN401
        return ready(started["apps"], tolerate=(apps_of(params["model"])["candidate"],))

    @task(task_id="canary_traffic")
    def traffic_task(
        checked: dict[str, Any],
        canary: dict[str, Any],
        params: Any = None,  # noqa: ANN401
    ) -> dict[str, Any] | None:
        settings = dagkit.settings()
        synthetic = params["synthetic_traffic"]
        if canary["deploy_failed"] or not (
            settings.canary_synthetic_traffic if synthetic is None else synthetic
        ):
            return None
        reg = dagkit.registry()
        spec = dagkit.spec_for_version(reg, reg.get_version(params["model"], params["version"]))
        return canary_traffic(
            ServeClient(settings.serve_url),
            dagkit.store(),
            spec,
            params["model"],
            checked["val_uri"],
            window_s=window_seconds(params, checked),
        )

    @task.sensor(task_id="observe", mode="reschedule", poke_interval=30, timeout=3 * 3600)
    def observe_task(
        checked: dict[str, Any],
        started: dict[str, Any],
        canary: dict[str, Any],
        params: Any = None,  # noqa: ANN401
    ) -> bool:
        end = datetime.fromisoformat(started["started_at"]) + timedelta(
            seconds=window_seconds(params, checked)
        )
        return bool(canary["deploy_failed"]) or datetime.now(UTC) >= end

    @task(task_id="judge")
    def judge_task(
        checked: dict[str, Any],
        canary: dict[str, Any],
        params: Any = None,  # noqa: ANN401
    ) -> dict[str, Any]:
        settings = dagkit.settings()
        if settings.prometheus_url is None and not canary["deploy_failed"]:
            raise RuntimeError("MODELPORT_PROMETHEUS_URL is not set: the canary cannot be judged")
        return judge(
            dagkit.registry(),
            PrometheusClient(settings.prometheus_url or ""),
            params["model"],
            params["version"],
            deploy_failed=canary["deploy_failed"],
            has_champion=checked["has_champion"],
            window_s=window_seconds(params, checked),
            now=datetime.now(UTC),
            pushgateway_url=settings.pushgateway_url,
        )

    @task.branch(task_id="branch")
    def branch_task(verdict: dict[str, Any]) -> str:
        return branch_choice(verdict)

    def serve_change(task_id: str, change: Any) -> Any:  # noqa: ANN401 - decorated task
        @task(task_id=task_id, pool=Pool.SERVE_CONFIG)
        def run(params: Any = None) -> list[str]:  # noqa: ANN401
            return list(change(dagkit.registry(), dagkit.kube(), dagkit.settings(), params["model"]))

        return run

    def wait(task_id: str) -> Any:  # noqa: ANN401 - decorated sensor
        @task.sensor(task_id=task_id, mode="reschedule", poke_interval=15, timeout=900)
        def poke(apps: list[str]) -> Any:  # noqa: ANN401
            return ready(apps)

        return poke

    @task(task_id="promote_aliases")
    def promote_task(params: Any = None) -> None:  # noqa: ANN401
        promote_aliases(
            dagkit.registry(), params["model"], params["version"], dagkit.settings().pushgateway_url
        )

    @task(task_id="mark_rejected")
    def reject_task(params: Any = None) -> None:  # noqa: ANN401
        mark_rejected(
            dagkit.registry(), params["model"], params["version"], dagkit.settings().pushgateway_url
        )

    checked = check_task()
    started = start_task(checked)
    canary = wait_canary_task(started)
    traffic = traffic_task(checked, canary)
    observed = observe_task(checked, started, canary)
    traffic >> observed
    verdict = judge_task(checked, canary)
    observed >> verdict
    chosen = branch_task(verdict)

    routed = serve_change("route_all_to_candidate", dagkit.route_all_to_candidate)()
    chosen >> routed
    promoted = promote_task()
    wait("wait_route")(routed) >> promoted
    updated = serve_change("update_champion_app", lambda reg, k, s, _model: apply_now(reg, k, s))()
    promoted >> updated
    retired = serve_change("retire_candidate_app", dagkit.clear_serve_candidate)()
    wait("wait_champion_ready")(updated) >> retired
    wait("wait_final")(retired)

    removed = serve_change("remove_candidate_app", dagkit.clear_serve_candidate)()
    chosen >> removed
    rejected = reject_task()
    removed >> rejected
    rejected >> wait("wait_after_rollback")(removed)


DAG = deploy_candidate()
