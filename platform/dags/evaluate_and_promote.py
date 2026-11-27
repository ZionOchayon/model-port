"""evaluate_and_promote: evaluate a new version and the champion on the same validation set, then gate (FR-7).

Both evaluations run on the shared RayCluster. A rejection is an outcome, not a failure: the run succeeds
and the reasons are stored on the version.
"""

import hashlib
import json
import tempfile
from collections.abc import Mapping, Sequence
from datetime import timedelta
from pathlib import Path
from typing import Any, Protocol

from airflow.providers.standard.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.sdk import Param, dag, task
from airflow.sdk.exceptions import AirflowSkipException

from modelport.lifecycle import ModelState, plan_gate
from modelport.naming import EXPERIMENT_EVALUATION, Tag, shared_job_id
from modelport.registry import ArtifactStore, ModelRegistry
from modelport.spec import ModelSpec
from modelport_platform import dagkit
from modelport_platform.dagkit import Pool
from modelport_platform.gate.core import EvalReport, GatePolicy, decide
from modelport_platform.metrics.push import push_aliases, push_gate_result, push_job_timing
from modelport_platform.ray.jobs import SharedClusterJobs
from modelport_platform.settings import PlatformSettings

EVAL_TIMEOUT_S = 1800


class Registry(ModelRegistry, ArtifactStore, Protocol):
    """Registry and artifacts, plus the find-or-create of runs."""

    def ensure_run(
        self, experiment: str, run_name: str, key_tag: Tag, tags: Mapping[str, str], params: Mapping[str, Any]
    ) -> str:
        """Find or create a run by tag."""


def _eval_run(registry: Registry, model: str, version: str, val_uri: str, dag_run_id: str) -> str:
    key = f"{model}/{version}/{val_uri}/{dag_run_id}"
    tags = {Tag.EVAL_KEY: key, Tag.MODEL: model}
    return registry.ensure_run(EXPERIMENT_EVALUATION, f"{model}-v{version}", Tag.EVAL_KEY, tags, {})


def _pip(tags: Mapping[str, str]) -> list[str]:
    return list(json.loads(tags.get(Tag.RUNTIME_PIP) or "[]"))


def prepare(conf: Mapping[str, Any], dag_run_id: str, registry: Registry, tmp: Path) -> dict[str, Any]:
    """Pick the validation set (the champion's, so both are judged on the same data) and the eval runs.

    Raises:
        AirflowSkipException: the version was already decided (a rerun is safe).
    """
    model, version = conf["model"], conf["version"]
    info = registry.get_version(model, version)
    if info.state is not ModelState.REGISTERED:
        raise AirflowSkipException(f"v{version} is already {info.state}")
    snapshot = registry.snapshot(model)
    champion = snapshot.champion if snapshot.champion not in (None, version) else None
    champion_info = registry.get_version(model, champion) if champion else None
    val_uri = (champion_info or info).tags[Tag.VAL_URI]
    spec = dagkit.spec_for_version(registry, info)
    runtime_pip = {version: _pip(info.tags)}
    if champion_info is not None:
        runtime_pip[champion_info.version] = _pip(champion_info.tags)
    return {
        "version": version,
        "champion": champion,
        "val_uri": val_uri,
        "candidate_run": _eval_run(registry, model, version, val_uri, dag_run_id),
        "champion_run": _eval_run(registry, model, champion, val_uri, dag_run_id) if champion else None,
        "runtime_pip": runtime_pip,
        "cpus": spec.resources.batch.cpus_per_task,
        "auto_deploy": spec.serving.auto_deploy,
    }


def run_evaluation(
    jobs: SharedClusterJobs,
    settings: PlatformSettings,
    *,
    model: str,
    version: str,
    val_uri: str,
    report_run: str,
    runtime_pip: Sequence[str],
    cpus: float,
    dag_run_id: str,
) -> None:
    """Submit (or adopt) the evaluation job of one version, wait for it and push its timing."""
    key = f"v{version}-{hashlib.sha256(dag_run_id.encode()).hexdigest()[:8]}"
    submission_id = shared_job_id("eval", model, key)
    runtime_env: dict[str, Any] = {
        # non-secret settings only: credentials come from the pods' Secret
        "env_vars": {
            "MLFLOW_TRACKING_URI": settings.mlflow_uri,
            "MODELPORT_S3_ENDPOINT": settings.s3_endpoint,
            "AWS_ENDPOINT_URL": settings.s3_endpoint,
        },
    }
    if runtime_pip:
        runtime_env["pip"] = {"packages": list(runtime_pip), "pip_check": False}
    jobs.submit_idempotent(
        submission_id=submission_id,
        entrypoint=(
            f"python -m modelport_platform.gate.evaluate --model {model} --version {version}"
            f" --val-uri {val_uri} --report-run-id {report_run}"
        ),
        runtime_env=runtime_env,
        entrypoint_num_cpus=cpus,
        metadata={"model": model, "version": version, "kind": "eval"},
    )
    timings = jobs.wait(submission_id, timeout_s=EVAL_TIMEOUT_S)
    push_job_timing(
        settings.pushgateway_url,
        model=model,
        pattern="shared",
        kind="eval",
        startup_s=timings.startup_s,
        duration_s=timings.duration_s,
        cpu_s=None,
    )


def decide_version(
    registry: Registry, model: str, prepared: Mapping[str, Any], *, pushgateway_url: str | None
) -> dict[str, Any]:
    """Apply the gate, record the result on the version and move the aliases."""
    version = prepared["version"]
    candidate = EvalReport.from_json(registry.read_json(prepared["candidate_run"], "gate/report.json"))
    champion = (
        EvalReport.from_json(registry.read_json(prepared["champion_run"], "gate/report.json"))
        if prepared["champion_run"]
        else None
    )
    spec: ModelSpec = dagkit.spec_for_version(registry, registry.get_version(model, version))
    decision = decide(candidate, champion, GatePolicy.from_spec(spec.evaluation))
    if champion is not None:  # `modelport status` shows both reports of this gate run side by side
        registry.log_json(prepared["candidate_run"], "gate/champion_report.json", champion.to_json())
    reasons = list(decision.reasons)
    registry.set_version_tags(
        model,
        version,
        {
            Tag.GATE_RESULT: "passed" if decision.passed else "failed",
            Tag.GATE_REASONS: json.dumps(reasons),
            Tag.GATE_EVAL_RUN_ID: prepared["candidate_run"],
        },
    )
    registry.apply_plan(model, plan_gate(registry.snapshot(model), version, decision.passed))
    push_gate_result(pushgateway_url, model, decision.passed)
    push_aliases(pushgateway_url, model, registry.snapshot(model))
    return {"passed": decision.passed, "reasons": reasons, "auto_deploy": prepared["auto_deploy"]}


def should_deploy(result: Mapping[str, Any]) -> bool:
    """Deploy a passed version unless serving.auto_deploy is off."""
    return bool(result["passed"] and result["auto_deploy"])


@dag(
    dag_id="evaluate_and_promote",
    schedule=None,
    max_active_runs=2,
    catchup=False,
    default_args=dagkit.DEFAULT_ARGS,
    tags=[*dagkit.TAGS, "gate"],
    doc_md=__doc__,
    params={
        "model": Param(type="string", minLength=1),
        "version": Param(type="string", pattern=r"^[0-9]+$"),
    },
)
def evaluate_and_promote() -> None:
    """Prepare -> evaluate candidate and champion -> decide -> deploy if passed."""

    @task(task_id="prepare")
    def prepare_task(params: Any = None, run_id: Any = None) -> dict[str, Any]:  # noqa: ANN401
        with tempfile.TemporaryDirectory() as tmp:
            return prepare(params, run_id, dagkit.registry(), Path(tmp))

    def _evaluate(prepared: dict[str, Any], version: str | None, report_run: str | None, run_id: str) -> None:
        if version is None or report_run is None:
            raise AirflowSkipException("no champion to evaluate")
        run_evaluation(
            dagkit.jobs(),
            dagkit.settings(),
            model=prepared["model"],
            version=version,
            val_uri=prepared["val_uri"],
            report_run=report_run,
            runtime_pip=prepared["runtime_pip"][version],
            cpus=prepared["cpus"],
            dag_run_id=run_id,
        )

    @task(pool=Pool.SHARED_CLUSTER, execution_timeout=timedelta(seconds=EVAL_TIMEOUT_S))
    def evaluate_candidate(prepared: dict[str, Any], params: Any = None, run_id: Any = None) -> None:  # noqa: ANN401
        _evaluate(
            {**prepared, "model": params["model"]}, prepared["version"], prepared["candidate_run"], run_id
        )

    @task(pool=Pool.SHARED_CLUSTER, execution_timeout=timedelta(seconds=EVAL_TIMEOUT_S))
    def evaluate_champion(prepared: dict[str, Any], params: Any = None, run_id: Any = None) -> None:  # noqa: ANN401
        _evaluate(
            {**prepared, "model": params["model"]}, prepared["champion"], prepared["champion_run"], run_id
        )

    @task(task_id="decide", trigger_rule="none_failed_min_one_success")  # champion eval may be skipped
    def decide_task(prepared: dict[str, Any], params: Any = None) -> dict[str, Any]:  # noqa: ANN401
        return decide_version(
            dagkit.registry(), params["model"], prepared, pushgateway_url=dagkit.settings().pushgateway_url
        )

    @task.short_circuit(task_id="should_deploy")
    def should_deploy_task(result: dict[str, Any]) -> bool:
        return should_deploy(result)

    prepared = prepare_task()
    result = decide_task(prepared)
    [evaluate_candidate(prepared), evaluate_champion(prepared)] >> result
    should_deploy_task(result) >> TriggerDagRunOperator(
        task_id="trigger_deploy",
        trigger_dag_id="deploy_candidate",
        trigger_run_id="deploy__{{ params.model }}__v{{ params.version }}",
        conf={"model": "{{ params.model }}", "version": "{{ params.version }}"},
        skip_when_already_exists=True,
    )


DAG = evaluate_and_promote()
