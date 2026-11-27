"""Check a modelport stack from its artifacts (MLflow, S3, Airflow REST, KubeRay objects), not exit codes.

Usage: uv run python scripts/verify.py --profile smoke|full [--model NAME] [--checks V-04,V-06] [--stage v0.1]
Prints PASS/FAIL per check; exit 1 if any check fails. Needs `make up` and `make ui`.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
import time
import zipfile
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
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


CHECKS: dict[str, Callable[[Context], None]] = {
    "V-01": v01,
    "V-02": v02,
    "V-04": v04,
    "V-05": v05,
    "V-06": v06,
    "V-07a": v07a,
    "V-07b": v07b,
    "V-08": v08,
}


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
