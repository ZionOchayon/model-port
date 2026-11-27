"""serve_sync: manual promote and rollback with the zero-downtime swap (FR-10).

`modelport promote` and `modelport rollback` start it. The target version is first served as a second
app with no traffic, then gets all of it; the champion app is updated while idle; the second app is
removed once the champion app runs the target. A model that is not served yet skips the staging.
"""

from typing import Any

from airflow.sdk import Param, dag, task
from airflow.sdk.exceptions import AirflowFailException

from modelport.errors import ModelportError
from modelport.lifecycle import Event, ModelState, plan_promote, plan_rollback
from modelport.logs import get_logger
from modelport.naming import MODEL_NAME_PATTERN, Tag
from modelport_platform import dagkit
from modelport_platform.dagkit import Pool, ServeRegistry
from modelport_platform.metrics.push import push_aliases

_log = get_logger(__name__)


def _refuse(what: str, fix: str) -> AirflowFailException:
    return AirflowFailException(f"{what}\n  How to fix: {fix}")


def plan(reg: ServeRegistry, model: str, action: str, version: str | None) -> dict[str, Any]:
    """The target version and whether the model is served; refuses what the CLI would refuse.

    Raises:
        AirflowFailException: a canary is running, the version did not pass the gate (promote), or there is
            no previous version (rollback).
    """
    if reg.model_tags(model).get(Tag.SERVE_CANDIDATE_VERSION):
        raise _refuse(
            f"a canary is running for {model}", "Wait for its deploy_candidate run to finish, then try again."
        )
    snapshot = reg.snapshot(model)
    if action == "rollback":
        if snapshot.previous is None:
            raise _refuse(f"{model} has no previous version to roll back to", "Promote a version instead.")
        target = snapshot.previous
    else:
        if version is None:
            raise _refuse("promote needs a version", f"Run `modelport promote {model} <version>`.")
        info = reg.get_version(model, version)
        if info.tags.get(Tag.GATE_RESULT) != "passed" or info.state is ModelState.REJECTED:
            raise _refuse(
                f"{model} v{version} did not pass the gate", f"`modelport status {model}` shows the reasons."
            )
        if snapshot.champion == version:
            raise _refuse(f"{model} v{version} is already the champion", "Nothing to do.")
        target = version
    return {"target": target, "served": snapshot.champion is not None}


def staging_choice(planned: dict[str, Any]) -> str:
    """Stage the target beside a served champion; an unserved model goes straight to the aliases."""
    return "stage_target" if planned["served"] else "apply_aliases"


def apply_aliases(
    reg: ServeRegistry, model: str, action: str, target: str, pushgateway_url: str | None
) -> None:
    """Move the aliases: the target becomes champion (promote), or previous comes back (rollback)."""
    snapshot = reg.snapshot(model)
    try:
        if action == "rollback":
            plan_ = plan_rollback(snapshot)
        else:
            state = reg.get_version(model, target).state
            plan_ = plan_promote(snapshot, target, event=Event.MANUAL_PROMOTE, current_state=state)
    except ModelportError as err:
        raise _refuse(err.what, err.fix) from err
    reg.apply_plan(model, plan_)
    push_aliases(pushgateway_url, model, reg.snapshot(model))
    _log.info("serve_sync.aliases", model=model, action=action, target=target)


@dag(
    dag_id="serve_sync",
    schedule=None,
    max_active_runs=1,
    catchup=False,
    default_args=dagkit.DEFAULT_ARGS,
    tags=[*dagkit.TAGS, "serve"],
    doc_md=__doc__,
    params={
        "model": Param(type="string", pattern=MODEL_NAME_PATTERN),
        "action": Param(type="string", enum=["promote", "rollback"]),
        "version": Param(None, type=["null", "string"], pattern=r"^[0-9]+$"),
    },
)
def serve_sync() -> None:
    """Plan, stage, swap, move the aliases."""

    @task(task_id="plan")
    def plan_task(params: Any = None) -> dict[str, Any]:  # noqa: ANN401
        return plan(dagkit.registry(), params["model"], params["action"], params["version"])

    @task.branch(task_id="branch")
    def branch_task(planned: dict[str, Any]) -> str:
        return staging_choice(planned)

    @task(task_id="stage_target", pool=Pool.SERVE_CONFIG)
    def stage_task(planned: dict[str, Any], params: Any = None) -> list[str]:  # noqa: ANN401
        reg, settings = dagkit.registry(), dagkit.settings()
        return dagkit.serve_target(reg, dagkit.kube(), settings, params["model"], planned["target"], 0.0)

    def wait(task_id: str) -> Any:  # noqa: ANN401 - decorated sensor
        @task.sensor(task_id=task_id, mode="reschedule", poke_interval=15, timeout=900)
        def poke(apps: list[str]) -> Any:  # noqa: ANN401
            return dagkit.rayservice_ready_poke(dagkit.kube(), apps)

        return poke

    def serve_change(task_id: str, change: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        @task(task_id=task_id, pool=Pool.SERVE_CONFIG, **kwargs)
        def run(params: Any = None) -> list[str]:  # noqa: ANN401
            return list(change(dagkit.registry(), dagkit.kube(), dagkit.settings(), params["model"]))

        return run

    @task(task_id="apply_aliases", trigger_rule="none_failed_min_one_success")  # after staging, or skipped
    def aliases_task(planned: dict[str, Any], params: Any = None) -> None:  # noqa: ANN401
        apply_aliases(
            dagkit.registry(),
            params["model"],
            params["action"],
            planned["target"],
            dagkit.settings().pushgateway_url,
        )

    planned = plan_task()
    chosen = branch_task(planned)
    staged = stage_task(planned)
    chosen >> staged
    routed = serve_change("route_all_to_target", dagkit.route_all_to_candidate)()
    wait("wait_staged")(staged) >> routed
    aliases = aliases_task(planned)
    chosen >> aliases
    wait("wait_routed")(routed) >> aliases
    updated = serve_change("update_champion_app", lambda reg, k, s, _m: dagkit.apply_serve_state(reg, k, s))()
    aliases >> updated
    retired = serve_change("retire_target_app", dagkit.clear_serve_candidate)()
    wait("wait_champion_ready")(updated) >> retired
    wait("wait_final")(retired)


DAG = serve_sync()
