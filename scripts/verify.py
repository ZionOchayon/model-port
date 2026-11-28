"""Check a modelport stack from its artifacts (MLflow, S3, Airflow REST, KubeRay objects), not exit codes.

Usage: uv run python scripts/verify.py --profile smoke|full [--model NAME] [--checks V-04,V-06] [--stage v0.1]
Prints PASS/FAIL per check; exit 1 if any check fails. Needs `make up` and `make ui`.
"""

from __future__ import annotations

import argparse
import io
import json
import re
import subprocess
import sys
import tempfile
import time
import zipfile
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

ROOT = Path(__file__).resolve().parents[1]
PROMETHEUS_URL: Final = "http://localhost:9090"  # make ui
sys.path.insert(0, str(Path(__file__).resolve().parent))

STAGE_CHECKS: dict[str, tuple[str, ...]] = {
    "v0.1": ("V-01", "V-02", "V-04", "V-05", "V-05d", "V-06", "V-07a", "V-07b", "V-08"),
    "v0.2": ("V-09", "V-10", "V-11", "V-12"),
    "v0.3": ("V-14", "V-16"),
}
STAGES = tuple(STAGE_CHECKS)
LINEAGE_TAGS = (
    "modelport.dataset",
    "modelport.dataset_version",
    "modelport.dataset_sha256",
    "modelport.val_uri",
    "modelport.code_sha256",
    "modelport.spec_sha256",
    "modelport.submission_run_id",
    "modelport.train_run_id",
    "modelport.train_rayjob",
    "modelport.image",
    "modelport.runtime_pip",
    "modelport.framework",
)
REPORT_KEYS = (
    "model",
    "version",
    "val_uri",
    "n_samples",
    "metric_name",
    "metric",
    "per_class",
    "latency_ms_median",
    "latency_batch_size",
    "created_at",
)
TRAIN_RUN_ARTIFACTS = ("spec/model.yaml", "drift/baseline.json", "train/runtime_pip.json")
CLUSTER_GONE_WAIT_S = 120


# ---- pure helpers (unit-tested) ----


def checks_for_stage(stage: str) -> list[str]:
    """Every check up to and including `stage` (stages are cumulative)."""
    return [c for s in STAGES[: STAGES.index(stage) + 1] for c in STAGE_CHECKS[s]]


def parse_checks(text: str) -> list[str]:
    """`V-01,V-06` -> ["V-01", "V-06"]; an unknown ID exits with a message."""
    known = {c for checks in STAGE_CHECKS.values() for c in checks}
    wanted = [c.strip() for c in text.split(",") if c.strip()]
    unknown = [c for c in wanted if c not in known]
    if unknown:
        raise SystemExit(f"verify: unknown checks {', '.join(unknown)} (known: {', '.join(sorted(known))})")
    return wanted


def default_stage(changelog: str) -> str:
    """The stage of the latest released `## [0.N.0]` heading; v0.1 before the first release."""
    released = re.findall(r"^## \[0\.(\d+)\.0\]", changelog, re.MULTILINE)
    return f"v0.{max(int(n) for n in released)}" if released else "v0.1"


def zip_extras(entries: Iterable[str], folder_files: set[str]) -> list[str]:
    """Zip entries that are not `model/<a file of the model folder>`."""
    return sorted(
        e for e in entries if not (e.startswith("model/") and e.removeprefix("model/") in folder_files)
    )


def missing_steps(steps: Iterable[int], *, epochs: int) -> list[int]:
    """Epoch steps 0..epochs-1 that were not logged."""
    return sorted(set(range(epochs)) - set(steps))


def resume_problems(steps: Sequence[int], *, epochs: int, resumed_from: Sequence[float]) -> list[str]:
    """Evidence that a killed run resumed: every epoch logged, epoch 0 once, some epoch resumed."""
    problems = []
    missing = missing_steps(steps, epochs=epochs)
    if missing:
        problems.append(f"epoch steps {missing} missing")
    if list(steps).count(0) > 1:
        problems.append(f"epoch step 0 logged {list(steps).count(0)} times (training restarted, not resumed)")
    if not any(v > 0 for v in resumed_from):
        problems.append("no epoch resumed from a checkpoint (resumed_from stayed 0)")
    return problems


def missing_lineage(tags: Mapping[str, str]) -> list[str]:
    """Lineage tag keys that are absent or empty."""
    return [k for k in LINEAGE_TAGS if not tags.get(k)]


def report_missing_keys(report: Mapping[str, Any]) -> list[str]:
    """Keys of `gate/report.json` that are absent."""
    return [k for k in REPORT_KEYS if k not in report]


def report_results(results: Sequence[tuple[str, str | None]]) -> int:
    """Print PASS/FAIL lines; 1 if anything failed."""
    for check, problem in results:
        print(f"PASS {check}" if problem is None else f"FAIL {check}: {problem}")
    return 1 if any(problem is not None for _, problem in results) else 0


def model_folder(model: str) -> tuple[Path, str]:
    """The example folder and spec file whose `name:` is `model`."""
    for spec in sorted(ROOT.glob("examples/models/*/model*.yaml")):
        if re.search(rf"^name:\s*{re.escape(model)}\s*$", spec.read_text(encoding="utf-8"), re.MULTILINE):
            return spec.parent, spec.name
    raise SystemExit(f"verify: no example spec has name '{model}'")


def folder_files(folder: Path) -> set[str]:
    """Files of a model folder as the code zip names them (relative, no caches)."""
    return {
        p.relative_to(folder).as_posix()
        for p in folder.rglob("*")
        if p.is_file() and "__pycache__" not in p.parts and not p.name.endswith(".pyc")
    }


def pip_by_app(serve_config: str) -> dict[str, tuple[str, ...]]:
    """Each Serve app's `runtime_env.pip` packages, from the RayService's `serveConfigV2` YAML."""
    import yaml

    apps = yaml.safe_load(serve_config).get("applications") or []
    return {
        a["name"]: tuple(((a.get("runtime_env") or {}).get("pip") or {}).get("packages") or ()) for a in apps
    }


# ---- checks against the stack ----


class CheckFailed(Exception):
    """One check's reason for failing."""


def require(condition: object, problem: str) -> None:
    """Fail the current check with `problem` unless `condition` holds."""
    if not condition:
        raise CheckFailed(problem)


@dataclass
class Context:
    """Adapters for the checks, built once."""

    model: str
    registry: Any  # MlflowRegistry
    airflow: Any  # AirflowClient
    store: Any  # S3DatasetStore
    profile: str = "smoke"

    def versions(self) -> list[Any]:
        """Every version of the model (MLflow ModelVersion objects)."""
        return list(self.registry.client.search_model_versions(f"name = '{self.model}'"))

    def spec(self, info: Any) -> Any:  # noqa: ANN401
        """The spec a version was trained with."""
        from modelport.naming import Tag
        from modelport.spec import load_spec

        with tempfile.TemporaryDirectory() as tmp:
            return load_spec(
                self.registry.download(info.tags[Tag.TRAIN_RUN_ID], "spec/model.yaml", Path(tmp))
            )

    def submit_run(self) -> str:
        """The train_model run of the latest submission."""
        from modelport.naming import Tag, submit_run_id

        submission = self.registry.model_tags(self.model).get(Tag.LATEST_SUBMISSION_RUN_ID)
        require(submission, "the model has no submission")
        key = self.registry.run_tags(submission).get(Tag.SUBMISSION_KEY)
        require(key, f"submission run {submission} has no submission key")
        return submit_run_id(self.model, key)

    def good_version(self) -> Any:  # noqa: ANN401
        """The version the candidate (or, from v0.2, the champion) alias points to."""
        from modelport.naming import Alias

        info = self.registry.resolve(self.model, Alias.CANDIDATE) or self.registry.resolve(
            self.model, Alias.CHAMPION
        )
        require(info, f"{self.model} has no candidate or champion")
        return info


def v01(ctx: Context) -> None:
    """Every pod Ready, shared RayCluster ready, all DAGs loaded without import errors."""
    import wait_ready

    problems = wait_ready.check_once()
    require(not problems, "; ".join(problems))


def v02(ctx: Context) -> None:
    """The model reached candidate/champion from its model folder alone."""
    from modelport.naming import Tag

    ctx.good_version()
    submission = ctx.registry.model_tags(ctx.model)[Tag.LATEST_SUBMISSION_RUN_ID]
    folder, _ = model_folder(ctx.model)
    with tempfile.TemporaryDirectory() as tmp:
        archive = ctx.registry.download(submission, "code/code.zip", Path(tmp))
        with zipfile.ZipFile(archive) as zf:
            extras = zip_extras(zf.namelist(), folder_files(folder))
    require(not extras, f"code.zip has files outside {folder.relative_to(ROOT)}: {', '.join(extras)}")


def v04(ctx: Context) -> None:
    """Training ran as a RayJob that succeeded on >= 2 workers, and its cluster is gone."""
    from modelport.naming import Tag

    run_id = ctx.submit_run()
    timing = ctx.airflow.xcom("train_model", run_id, "wait_rayjob")
    require(timing and timing.get("job_status") == "SUCCEEDED", f"wait_rayjob XCom of {run_id} is {timing}")
    info = ctx.good_version()
    params = ctx.registry.client.get_run(info.tags[Tag.TRAIN_RUN_ID]).data.params
    require(int(params.get("workers", 0)) >= 2, f"training run param workers={params.get('workers')}")
    deadline = time.monotonic() + CLUSTER_GONE_WAIT_S
    while True:
        out = _kubectl("get", "pods", "-n", "ray", "-l", "modelport.dev/kind=train", "-o", "name")
        if not out.strip():
            return
        require(time.monotonic() < deadline, f"training pods still exist: {out.split()}")
        time.sleep(10)


def v05(ctx: Context) -> None:
    """A checkpoint was registered at every epoch."""
    from modelport.naming import Tag

    info = ctx.good_version()
    history = ctx.registry.client.get_metric_history(info.tags[Tag.TRAIN_RUN_ID], "checkpoint_saved")
    missing = missing_steps([m.step for m in history], epochs=ctx.spec(info).train.epochs)
    require(not missing, f"no checkpoint_saved at epoch steps {missing}")


def v06(ctx: Context) -> None:
    """Signature, lineage tags and the training run's artifacts of the newest version."""
    import mlflow

    from modelport.naming import Tag

    info = ctx.registry.latest_version(ctx.model)
    require(info, "no versions")
    spec = ctx.spec(info)
    signature = mlflow.models.get_model_info(f"models:/{ctx.model}/{info.version}").signature
    inputs = [c.name for c in signature.inputs.inputs]
    require(inputs == [c.name for c in spec.signature.inputs], f"signature inputs {inputs}")
    outputs = [c.name for c in signature.outputs.inputs]
    require(outputs == ["prediction", "confidence"], f"signature outputs {outputs}")
    missing = missing_lineage(info.tags)
    require(not missing, f"v{info.version} lineage tags empty: {', '.join(missing)}")
    run_id = info.tags[Tag.TRAIN_RUN_ID]
    present = {
        a.path for d in ("spec", "drift", "train") for a in ctx.registry.client.list_artifacts(run_id, d)
    }
    absent = [a for a in TRAIN_RUN_ARTIFACTS if a not in present]
    require(not absent, f"training run {run_id} lacks {', '.join(absent)}")


def v07a(ctx: Context) -> None:
    """The good version passed the gate and has a complete report."""
    from modelport.naming import Tag

    info = ctx.good_version()
    require(
        info.tags.get(Tag.GATE_RESULT) == "passed",
        f"v{info.version} gate result {info.tags.get(Tag.GATE_RESULT)}",
    )
    report = ctx.registry.read_json(info.tags[Tag.GATE_EVAL_RUN_ID], "gate/report.json")
    missing = report_missing_keys(report)
    require(not missing, f"gate/report.json lacks {', '.join(missing)}")


def v07b(ctx: Context) -> None:
    """A deliberately broken version was rejected with reasons and holds no alias."""
    rejected = [v for v in ctx.versions() if v.tags.get("modelport.state") == "rejected"]
    require(rejected, "no rejected version")
    for v in rejected:
        require(
            json.loads(v.tags.get("modelport.gate.reasons") or "[]"), f"v{v.version} rejected without reasons"
        )
        require(not v.aliases, f"rejected v{v.version} still has aliases {list(v.aliases)}")


def v08(ctx: Context) -> None:
    """Quarantine report for the corrupt file, dataset v2 and the retrain it triggered."""
    reports = [u for u in ctx.store.list("s3://quarantine/fashion-mnist/") if u.endswith("/report.json")]
    require(reports, "no quarantine report for fashion-mnist")
    failed = [u for u in reports if not json.loads(ctx.store.read_bytes(u)).get("passed", True)]
    require(failed, "no quarantine report with a failed check")
    require(
        ctx.store.exists("s3://datasets/fashion-mnist/v2/manifest.json"), "fashion-mnist v2 not committed"
    )
    run = ctx.airflow.get_run("train_model", f"retrain__{ctx.model}__fashion-mnist__v2")
    require(run and run.state == "success", f"retrain run is {run.state if run else 'absent'}")


SERVED_MODELS: Final = ("fashion-classifier", "covertype-classifier")


def v09(ctx: Context) -> None:
    """Both example models answer `modelport predict`, each app with its own runtime_env.pip."""
    for model in SERVED_MODELS:
        folder, _ = model_folder(model)
        example = folder / "example" / "input.parquet"
        out = subprocess.run(  # noqa: S603 - fixed command
            ["uv", "run", "modelport", "--json", "predict", model, str(example)],  # noqa: S607
            capture_output=True,
            text=True,
            check=False,
            cwd=ROOT,
        )
        require(
            out.returncode == 0, f"modelport predict {model} exited {out.returncode}: {out.stdout[-300:]}"
        )
        answer = json.loads(out.stdout)
        import pandas as pd

        rows = len(pd.read_parquet(example))
        require(
            len(answer["predictions"]) == rows,
            f"{model}: {len(answer['predictions'])} predictions for {rows} rows",
        )
    config = _kubectl(
        "get", "rayservice", "-n", "ray", "modelport-serve", "-o", "jsonpath={.spec.serveConfigV2}"
    )
    pips = pip_by_app(config)
    champions = [pips.get(f"{m}--champion") for m in SERVED_MODELS]
    require(
        all(p is not None for p in champions), f"champion apps missing from the RayService: {sorted(pips)}"
    )
    require(champions[0] != champions[1], f"both champion apps have runtime_env.pip {champions[0]}")


DRILL_KILL_WORKER = ROOT / ".data" / "drill-kill-worker.json"


def v05d(ctx: Context) -> None:
    """After bench/drills/kill_worker.sh: the run succeeded and resumed from epoch 0's checkpoint."""
    from modelport.naming import Tag

    require(
        DRILL_KILL_WORKER.is_file(), f"{DRILL_KILL_WORKER.relative_to(ROOT)} missing (run the drill first)"
    )
    drill = json.loads(DRILL_KILL_WORKER.read_text(encoding="utf-8"))
    run = ctx.airflow.get_run("train_model", drill["run_id"])
    require(
        run and run.state == "success", f"drill run {drill['run_id']} is {run.state if run else 'absent'}"
    )
    info = ctx.registry.find_version_by_tag(ctx.model, Tag.TRAIN_RAYJOB, drill["rayjob"])
    require(info, f"no version from RayJob {drill['rayjob']}")
    client, train_run = ctx.registry.client, info.tags[Tag.TRAIN_RUN_ID]
    saved = client.get_metric_history(train_run, "checkpoint_saved")
    resumed = client.get_metric_history(train_run, "resumed_from")
    first_zero = min((m.timestamp for m in saved if m.step == 0), default=None)
    require(first_zero is not None, "epoch 0's checkpoint was never registered")
    require(
        drill["killed_at_ms"] > first_zero, "the worker was killed before epoch 0's checkpoint was registered"
    )
    problems = resume_problems(
        [m.step for m in saved], epochs=int(drill["epochs"]), resumed_from=[m.value for m in resumed]
    )
    require(not problems, "; ".join(problems))


DRILL_ZERO_DOWNTIME = ROOT / ".data" / "drill-zero-downtime.json"
MIN_DRILL_REQUESTS: Final = 300


def zero_downtime_problems(drill: Mapping[str, Any]) -> list[str]:
    """What the zero-downtime drill's result breaks of FR-11: no failed request while the champion changed."""
    problems = []
    if drill["failed"]:
        problems.append(
            f"{drill['failed']} of {drill['sent']} requests failed ({drill.get('status_counts')})"
        )
    if drill["sent"] < MIN_DRILL_REQUESTS:
        problems.append(f"only {drill['sent']} requests sent (at least {MIN_DRILL_REQUESTS})")
    if drill["champion_before"] == drill["champion_after"]:
        problems.append(f"the champion stayed v{drill['champion_after']}")
    elif not drill["promoted_during_load"]:
        problems.append("the promotion finished after the load ended")
    return problems


def v11(ctx: Context) -> None:
    """After bench/drills/zero_downtime.sh: a promotion under load with no failed request (FR-11)."""
    require(
        DRILL_ZERO_DOWNTIME.is_file(),
        f"{DRILL_ZERO_DOWNTIME.relative_to(ROOT)} missing (run the drill first)",
    )
    problems = zero_downtime_problems(json.loads(DRILL_ZERO_DOWNTIME.read_text(encoding="utf-8")))
    require(not problems, "; ".join(problems))


DRILL_CANARY_FAILURE = ROOT / ".data" / "drill-canary-failure.json"


def canary_failure_problems(
    model: str,
    version: str,
    *,
    state: str,
    tags: Mapping[str, str],
    aliases: Mapping[str, str | None],
    champion_before: str,
    apps: Sequence[str],
    last_result: float | None,
) -> list[str]:
    """What breaks FR-10 after the canary failure drill: the faulty version must be rolled back cleanly.

    `last_result` is `modelport_canary_last_result{model}` in Prometheus (pushed by the judge).
    """
    from modelport.naming import Tag

    problems = []
    if state != "rejected":
        problems.append(f"v{version} is {state}, not rejected")
    reasons = json.loads(tags.get(Tag.CANARY_REASONS) or "[]")
    if tags.get(Tag.CANARY_RESULT) != "failed":
        problems.append(f"canary result is {tags.get(Tag.CANARY_RESULT)}, not failed")
    elif not any("error rate" in r for r in reasons):
        problems.append(f"no error-rate reason: {reasons}")
    held = [alias for alias, v in sorted(aliases.items()) if v == version]
    if held:
        problems.append(f"v{version} still has the alias {', '.join(held)}")
    if aliases.get("champion") != champion_before:
        problems.append(f"the champion changed from v{champion_before} to v{aliases.get('champion')}")
    if f"{model}--candidate" in apps:
        problems.append(f"the RayService still serves {model}--candidate")
    if last_result != 0:
        shown = "absent" if last_result is None else last_result
        problems.append(f"modelport_canary_last_result is {shown} in Prometheus, not 0")
    return problems


def v10(ctx: Context) -> None:
    """After bench/drills/canary_failure.sh: the faulty version is rejected and the champion kept serving."""
    require(
        DRILL_CANARY_FAILURE.is_file(),
        f"{DRILL_CANARY_FAILURE.relative_to(ROOT)} missing (run the drill first)",
    )
    drill = json.loads(DRILL_CANARY_FAILURE.read_text(encoding="utf-8"))
    model, version = drill["model"], drill["version"]
    run = ctx.airflow.get_run("deploy_candidate", drill["run_id"])
    require(
        run and run.state == "success", f"drill run {drill['run_id']} is {run.state if run else 'absent'}"
    )
    info = ctx.registry.get_version(model, version)
    snapshot = ctx.registry.snapshot(model)
    config = _kubectl(
        "get", "rayservice", "-n", "ray", "modelport-serve", "-o", "jsonpath={.spec.serveConfigV2}"
    )
    problems = canary_failure_problems(
        model,
        version,
        state=str(info.state),
        tags=info.tags,
        aliases={
            "champion": snapshot.champion,
            "previous": snapshot.previous,
            "candidate": snapshot.candidate,
        },
        champion_before=drill["champion_before"],
        apps=list(pip_by_app(config)),
        last_result=_prometheus().scalar(f'max(modelport_canary_last_result{{model="{model}"}})'),
    )
    require(not problems, "; ".join(problems))


BATCH_RUNS = (ROOT / ".data" / "v12-run1.json", ROOT / ".data" / "v12-run2.json")


def batch_rerun_problems(
    run1: Mapping[str, Any], run2: Mapping[str, Any], input_rows: Mapping[str, int]
) -> list[str]:
    """What two forced batch runs of one day break of FR-12: same predictions, one per input row."""
    problems = []
    for model, rows in input_rows.items():
        if model not in run1:
            problems.append(f"{model}: no predictions")
        elif run1[model] != run2.get(model):
            problems.append(f"{model}: the two runs wrote different predictions")
        elif run1[model]["rows"] != rows:
            problems.append(f"{model}: {run1[model]['rows']} predictions for {rows} input rows")
    return problems


def _input_rows(store: Any, model: str) -> int:  # noqa: ANN401 - S3DatasetStore
    """Input rows of the model's newest batch job (its `_SUCCESS.json` names the input)."""
    import pyarrow.parquet as pq

    markers = [u for u in store.list(f"s3://predictions/{model}/") if u.endswith("/_SUCCESS.json")]
    require(markers, f"{model}: no batch job finished")
    newest = max((json.loads(store.read_bytes(u)) for u in markers), key=lambda m: m["finished_at"])
    files = [u for u in store.list(newest["input_uri"]) if u.endswith(".parquet")]
    return sum(pq.ParquetFile(io.BytesIO(store.read_bytes(u))).metadata.num_rows for u in files)


def v12(ctx: Context) -> None:
    """After two forced batch_inference runs of one day: identical predictions, one per input row."""
    missing = [p.relative_to(ROOT).as_posix() for p in BATCH_RUNS if not p.is_file()]
    require(not missing, f"{', '.join(missing)} missing (run snapshot-predictions after each run)")
    run1, run2 = (json.loads(p.read_text(encoding="utf-8")) for p in BATCH_RUNS)
    inputs = {model: _input_rows(ctx.store, model) for model in SERVED_MODELS}
    problems = batch_rerun_problems(run1, run2, inputs)
    require(not problems, "; ".join(problems))


BENCH_REPS = 5


def bench_problems(root: Path) -> list[str]:
    """V-13 on a checkout: the newest bench CSV has BENCH_REPS rows per pattern and the table matches it."""
    sys.path.insert(0, str(ROOT / "bench"))
    import summarize

    raw = sorted((root / "docs/results/raw").glob("bench-*.csv"))
    if not raw:
        return ["no docs/results/raw/bench-*.csv (run the bench workflow and commit its results)"]
    counts = summarize.counts(summarize.load(raw[-1]))
    problems = [
        f"{raw[-1].name}: {pattern} has {counts.get(pattern, 0)} rows, not {BENCH_REPS}"
        for pattern in summarize.PATTERNS
        if counts.get(pattern, 0) != BENCH_REPS
    ]
    if summarize.main(["--root", str(root), "--check", str(root / "README.md")]) != 0:
        problems.append("docs/results/table.md or the README block is stale (python bench/summarize.py)")
    return problems


def v13(ctx: Context) -> None:
    """The FR-13 table: n = 5 per pattern, and README and table.md generated from the newest CSV."""
    problems = bench_problems(ROOT)
    require(not problems, "; ".join(problems))


def v14(ctx: Context) -> None:
    """FR-14: scripts/metrics_check.py passes (every target up, every source with data)."""
    import metrics_check

    require(metrics_check.main(["--profile", ctx.profile]) == 0, "metrics_check.py failed (problems above)")


CHECKS: dict[str, Callable[[Context], None]] = {
    "V-14": v14,
    "V-13": v13,
    "V-12": v12,
    "V-10": v10,
    "V-09": v09,
    "V-11": v11,
    "V-01": v01,
    "V-02": v02,
    "V-04": v04,
    "V-05": v05,
    "V-05d": v05d,
    "V-06": v06,
    "V-07a": v07a,
    "V-07b": v07b,
    "V-08": v08,
}


def _prometheus() -> Any:  # noqa: ANN401 - PrometheusClient
    from modelport_platform.canary.prometheus import PrometheusClient

    return PrometheusClient(PROMETHEUS_URL)


def _kubectl(*args: str) -> str:
    import wait_ready

    out = wait_ready.run(*args)
    require(out.returncode == 0, f"kubectl {' '.join(args)}: {out.stderr.strip()}")
    return out.stdout


def build_context(model: str) -> Context:
    """Real adapters through the port-forwards (`make ui`)."""
    import mlflow
    import ops

    from modelport.clients.airflow import AirflowClient
    from modelport.config import ClientSettings
    from modelport.registry import MlflowRegistry

    settings = ClientSettings.from_env()
    mlflow.set_tracking_uri(settings.mlflow_url)
    airflow = AirflowClient(settings.airflow_url, settings.airflow_username, settings.airflow_password)
    return Context(model, MlflowRegistry(settings.mlflow_url), airflow, ops.dataset_store())


def main(argv: Sequence[str] | None = None) -> int:
    """Run the requested (or the stage's) checks."""
    parser = argparse.ArgumentParser(description="verify a modelport stack from its artifacts")
    parser.add_argument("--profile", choices=("smoke", "full"), required=True)
    parser.add_argument("--model", default="fashion-classifier")
    parser.add_argument("--checks", help="comma-separated check IDs (default: every check of --stage)")
    parser.add_argument("--stage", choices=STAGES)
    args = parser.parse_args(argv)
    stage = args.stage or default_stage((ROOT / "CHANGELOG.md").read_text(encoding="utf-8"))
    wanted = parse_checks(args.checks) if args.checks else checks_for_stage(stage)
    ctx = build_context(args.model)
    ctx.profile = args.profile
    results: list[tuple[str, str | None]] = []
    for check in wanted:
        fn = CHECKS.get(check)
        if fn is None:
            results.append((check, "not implemented yet"))
            continue
        try:
            fn(ctx)
            results.append((check, None))
        except CheckFailed as err:
            results.append((check, str(err)))
        except Exception as err:  # a broken check is a failed check, with its reason
            results.append((check, f"{type(err).__name__}: {err}"))
    return report_results(results)


if __name__ == "__main__":
    sys.exit(main())
