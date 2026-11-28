"""`make bench` (FR-13): the same batch job as a RayJob (cluster per job) and on the shared cluster.

Per rep, one job at a time: wait for the shared workers to reach 0, `shared` (workers from zero), right
after it `shared-warm` (worker still up), wait for 0 again, `rayjob`. Startup = submitted -> the driver is
RUNNING, duration = RUNNING -> done (polled every second), CPU-seconds from Prometheus. Then the three
failure drills once each. Writes docs/results/raw/bench-<UTC>.csv, docs/results/drills.md, and runs
summarize.py. Needs `make up`, `make ui` and .modelport/credentials.env in the environment.

`--dry-run-check` builds every submission, manifest and query from the repository files and prints them.
"""

import argparse
import copy
import csv
import json
import math
import os
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any, Final

import yaml

from modelport.naming import batch_input_uri, prediction_uri
from modelport.spec import load_spec
from modelport_platform.batch.plan import batch_job
from modelport_platform.ray.manifests import SUBMITTER

ROOT: Final = Path(__file__).resolve().parents[1]
MODEL: Final = "covertype-classifier"
DATASET: Final = "covertype"
BENCH_DAY: Final = date(2000, 1, 1)  # never a real day: the daily batch_inference run never scores it
ROWS: Final = 50_000
COLUMNS: Final = ["rep", "pattern", "startup_s", "duration_s", "cpu_s", "job_id", "started_at_utc", "host"]
LABELS: Final = {"app.kubernetes.io/part-of": "modelport", "modelport.dev/kind": "bench"}
SHARED_PODS: Final = "modelport-shared-.*"
AFTER_END_S: Final = 30  # two 15 s scrapes, so the job's last CPU samples are in Prometheus
POLL_S: Final = 1.0
JOB_TIMEOUT_S: Final = 3600
IDLE_TIMEOUT_S: Final = 900
KUBECTL: Final = ["kubectl", "--context", "kind-modelport"]
PROMETHEUS_URL: Final = "http://localhost:9090"  # make ui
RAY_DASHBOARD_URL: Final = "http://localhost:8265"  # make ui
CLUSTER_DEFAULTS: Final = {  # what scripts/up.sh writes to .modelport/platform.env
    "MLFLOW_TRACKING_URI": "http://mlflow.platform.svc.cluster.local:5000",
    "MODELPORT_S3_ENDPOINT": "http://minio.platform.svc.cluster.local:9000",
    "MODELPORT_RAY_IMAGE": "__RAY_IMAGE__",
}
IMG: Final = "examples/models/image-classifier"
TAB: Final = "examples/models/covertype-classifier"
# (model, onboarding commands as in scripts/e2e.sh), skipped for a model that already has a champion
ONBOARD: Final = {
    "fashion-classifier": [
        ["prepare", "--dataset", "fashion-mnist", "--rows", "6000", "--upload"],
        ["ops", "wait-dataset", "fashion-mnist", "1", "--timeout", "900"],
        ["mp", "submit", IMG, "--spec", "model.fashion.yaml", "--wait", "--timeout", "3600"],
        ["ops", "wait-alias", "fashion-classifier", "champion", "--timeout", "2400"],
        ["ops", "wait-dag-run", "deploy_candidate", "deploy__fashion-classifier__v1", "--timeout", "1800"],
    ],
    MODEL: [
        ["prepare", "--dataset", DATASET, "--rows", "20000", "--upload"],
        ["ops", "wait-dataset", DATASET, "1", "--timeout", "900"],
        ["mp", "submit", TAB, "--wait", "--timeout", "3600"],
        ["ops", "wait-alias", MODEL, "champion", "--timeout", "2400"],
        ["ops", "wait-dag-run", "deploy_candidate", f"deploy__{MODEL}__v1", "--timeout", "1800"],
    ],
}
COMMANDS: Final = {
    "prepare": ["uv", "run", "python", "examples/datasets/prepare.py"],
    "ops": ["uv", "run", "python", "scripts/ops.py"],
    "mp": ["uv", "run", "modelport"],
}
# (drill script, its model, the verify.py check that judges it)
DRILLS: Final = [
    ("kill_worker", "fashion-classifier", "V-05d"),
    ("zero_downtime", "fashion-classifier", "V-11"),
    ("canary_failure", MODEL, "V-10"),
]


def rayjob_manifest(
    shared: Mapping[str, Any], *, name: str, job: Mapping[str, Any], image: str
) -> dict[str, Any]:
    """A RayJob whose cluster is the shared RayCluster's head and `batch` group, fixed at one worker."""
    cluster = copy.deepcopy(dict(shared["spec"]))
    cluster.pop("enableInTreeAutoscaling", None)
    cluster.pop("autoscalerOptions", None)
    group = cluster["workerGroupSpecs"][0]
    group.update(replicas=1, minReplicas=1, maxReplicas=1)
    cluster["workerGroupSpecs"] = [group]
    for template in (cluster["headGroupSpec"]["template"], group["template"]):
        template.setdefault("metadata", {})["labels"] = dict(LABELS)
    return {
        "apiVersion": "ray.io/v1",
        "kind": "RayJob",
        "metadata": {"name": name, "namespace": shared["metadata"]["namespace"], "labels": dict(LABELS)},
        "spec": {
            "entrypoint": job["entrypoint"],
            "entrypointNumCpus": job["entrypoint_num_cpus"],
            "runtimeEnvYAML": yaml.safe_dump(job["runtime_env"], sort_keys=True),
            "shutdownAfterJobFinishes": True,
            "ttlSecondsAfterFinished": 30,
            "activeDeadlineSeconds": JOB_TIMEOUT_S,
            "backoffLimit": 0,
            "submissionMode": "K8sJobMode",
            "submitterPodTemplate": {
                "spec": {
                    "restartPolicy": "Never",
                    "containers": [
                        {
                            "name": "ray-job-submitter",
                            "image": image,
                            "imagePullPolicy": "IfNotPresent",
                            "resources": SUBMITTER.resources(),
                        }
                    ],
                }
            },
            "rayClusterSpec": cluster,
        },
    }


def cpu_window(submitted: datetime, ended: datetime) -> tuple[datetime, int]:
    """(when to evaluate, window in s): AFTER_END_S after the end, back to the submission."""
    at = ended + timedelta(seconds=AFTER_END_S)
    return at, math.ceil((at - submitted).total_seconds())


def cpu_query(pods: str, window_s: int) -> str:
    """CPU-seconds of every container of the pods matching `pods` over the window."""
    return (
        f'sum(increase(container_cpu_usage_seconds_total{{namespace="ray", pod=~"{pods}",'
        f' container!=""}}[{window_s}s]))'
    )


@dataclass
class Run:
    """One measured job."""

    rep: int
    pattern: str
    job_id: str
    pods: str
    submitted: datetime
    running: datetime
    ended: datetime
    cpu_s: float | None = None

    def row(self, host: str) -> dict[str, Any]:
        """The CSV row."""
        return {
            "rep": self.rep,
            "pattern": self.pattern,
            "startup_s": round((self.running - self.submitted).total_seconds(), 2),
            "duration_s": round((self.ended - self.running).total_seconds(), 2),
            "cpu_s": "" if self.cpu_s is None else round(self.cpu_s, 2),
            "job_id": self.job_id,
            "started_at_utc": self.submitted.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "host": host,
        }


def now() -> datetime:
    """Current UTC time."""
    return datetime.now(UTC)


def poll_job(status: Callable[[], str], what: str) -> tuple[datetime, datetime]:
    """(first seen RUNNING, done): every POLL_S until SUCCEEDED; raises on failure or timeout."""
    deadline = time.monotonic() + JOB_TIMEOUT_S
    running: datetime | None = None
    while time.monotonic() < deadline:
        state = status()
        if state == "RUNNING" and running is None:
            running = now()
        if state == "SUCCEEDED":
            ended = now()
            return running or ended, ended
        if state in ("FAILED", "STOPPED"):
            raise RuntimeError(f"bench: {what} {state}")
        time.sleep(POLL_S)
    raise TimeoutError(f"bench: {what} did not finish in {JOB_TIMEOUT_S} s")


def kubectl(*args: str, stdin: str | None = None) -> str:
    """Run kubectl on the kind cluster only."""
    out = subprocess.run([*KUBECTL, *args], input=stdin, capture_output=True, text=True, check=True)  # noqa: S603
    return out.stdout


def shared_workers() -> int:
    """Worker pods of the shared cluster (any phase)."""
    out = kubectl(
        "get",
        "pods",
        "-n",
        "ray",
        "-o",
        "name",
        "-l",
        "ray.io/cluster=modelport-shared,ray.io/node-type=worker",
    )
    return len(out.split())


def wait_idle() -> None:
    """Until the shared cluster has scaled its workers to 0."""
    deadline = time.monotonic() + IDLE_TIMEOUT_S
    while shared_workers():
        if time.monotonic() >= deadline:
            raise TimeoutError(f"bench: shared workers not at 0 after {IDLE_TIMEOUT_S} s")
        time.sleep(5)


def run_shared(client: Any, job: Mapping[str, Any], rep: int, pattern: str) -> Run:  # noqa: ANN401
    """Submit to modelport-shared with the Jobs API and time it."""
    client.submit_job(
        entrypoint=job["entrypoint"],
        runtime_env=dict(job["runtime_env"]),
        submission_id=job["submission_id"],
        entrypoint_num_cpus=job["entrypoint_num_cpus"],
        metadata={**job["metadata"], "kind": "bench"},
    )
    submitted = now()
    sid = job["submission_id"]
    running, ended = poll_job(lambda: str(client.get_job_status(sid).value), sid)
    return Run(rep, pattern, sid, SHARED_PODS, submitted, running, ended)


def run_rayjob(manifest: Mapping[str, Any], rep: int) -> Run:
    """Create the RayJob and time it from creation to its job status."""
    name = manifest["metadata"]["name"]
    kubectl("create", "-f", "-", stdin=json.dumps(manifest))
    submitted = now()

    def status() -> str:
        found = json.loads(kubectl("get", "rayjob", name, "-n", "ray", "-o", "json")).get("status", {})
        if found.get("jobDeploymentStatus") == "Failed":
            return "FAILED"
        return str(found.get("jobStatus") or "PENDING")

    running, ended = poll_job(status, f"RayJob {name}")
    return Run(rep, "rayjob", name, f"{name}-.*", submitted, running, ended)


def add_cpu(runs: Sequence[Run], prometheus: Any) -> None:  # noqa: ANN401 - PrometheusClient
    """Fill each run's CPU-seconds, waiting until its evaluation time has passed."""
    for run in runs:
        at, window = cpu_window(run.submitted, run.ended)
        wait = (at - now()).total_seconds()
        if wait > 0:
            time.sleep(wait)
        run.cpu_s = prometheus.scalar(cpu_query(run.pods, window), at=at)
        if run.cpu_s is None:
            print(f"bench: no CPU samples for {run.job_id} ({run.pods})", file=sys.stderr)


def cluster_env() -> dict[str, str]:
    """The in-cluster settings from .modelport/platform.env (defaults as up.sh writes them)."""
    env = dict(CLUSTER_DEFAULTS)
    path = ROOT / ".modelport/platform.env"
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            key, sep, value = line.partition("=")
            if sep and not line.startswith("#"):
                env[key.strip()] = value.strip()
    return env


def jobs_for(
    item: Mapping[str, Any], env: Mapping[str, str], stamp: str, rep: int
) -> dict[str, dict[str, Any]]:
    """The three submissions of one rep, each with its own ID."""
    return {
        pattern: batch_job(
            item,
            force=True,
            mlflow_uri=env["MLFLOW_TRACKING_URI"],
            s3_endpoint=env["MODELPORT_S3_ENDPOINT"],
            key=f"bench-{stamp}-{pattern}-{rep}",
        )
        for pattern in ("shared", "shared-warm", "rayjob")
    }


def rayjob_name(stamp: str, rep: int) -> str:
    """A short, lower-case Kubernetes name."""
    return f"bench-{stamp.lower()}-{rep}"


def dry_run(reps: int) -> int:
    """Print what a run of `reps` would submit and query, from the repository files only."""
    shared = yaml.safe_load(
        (ROOT / "deploy/manifests/base/shared-raycluster.yaml").read_text(encoding="utf-8")
    )
    spec = load_spec(ROOT / TAB / "model.yaml")
    item = {
        "model": MODEL,
        "version": "1",
        "day": BENCH_DAY.isoformat(),
        "input_uri": batch_input_uri(DATASET, BENCH_DAY),
        "output_uri": prediction_uri(MODEL, BENCH_DAY),
        "runtime_pip": list(spec.runtime.pip),
        "batch_size": spec.resources.batch.batch_size,
        "num_cpus": spec.resources.batch.cpus_per_task,
    }
    env, stamp = cluster_env(), "dryrun"
    for rep in range(1, reps + 1):
        jobs = jobs_for(item, env, stamp, rep)
        for pattern in ("shared", "shared-warm"):
            print(f"{pattern} rep {rep}: {json.dumps(jobs[pattern], sort_keys=True)}")
        manifest = rayjob_manifest(
            shared, name=rayjob_name(stamp, rep), job=jobs["rayjob"], image=env["MODELPORT_RAY_IMAGE"]
        )
        print(f"rayjob rep {rep}:\n{yaml.safe_dump(manifest, sort_keys=False)}")
    print(f"cpu query: {cpu_query(SHARED_PODS, 120)}")
    return 0


def sh(*cmd: str) -> int:
    """Run a command from the repository root, output passed through."""
    print(f"bench: $ {' '.join(cmd)}", flush=True)
    return subprocess.run(list(cmd), cwd=ROOT, check=False).returncode  # noqa: S603 - our own commands


def onboard() -> None:
    """Both example models served (as e2e.sh does), then the bench day's input."""
    from modelport.config import ClientSettings
    from modelport.naming import Alias
    from modelport.registry import MlflowRegistry

    registry = MlflowRegistry(ClientSettings.from_env().mlflow_url)
    for model, steps in ONBOARD.items():
        if registry.resolve(model, Alias.CHAMPION) is not None:
            print(f"bench: {model} has a champion, onboarding skipped")
            continue
        for tool, *args in steps:
            if sh(*COMMANDS[tool], *args) != 0:
                raise RuntimeError(f"bench: onboarding {model} failed at {tool} {' '.join(args)}")
    sh("bash", "scripts/ui.sh")  # the Serve forward exists only once a model is served
    prepare = ["--batch-input", "--dataset", DATASET, "--day", BENCH_DAY.isoformat(), "--rows", str(ROWS)]
    if sh(*COMMANDS["prepare"], *prepare, "--upload") != 0:
        raise RuntimeError("bench: writing the bench input failed")


def host_label() -> str:
    """The first line of host.txt (the bench workflow writes it), else the CPU count."""
    path = ROOT / "host.txt"
    if path.is_file():
        return path.read_text(encoding="utf-8").splitlines()[0].strip()
    return f"{os.cpu_count()} vCPU"


def write_csv(path: Path, runs: Sequence[Run], host: str) -> None:
    """All rows so far (rewritten after every rep, so a failed run keeps what it measured)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(run.row(host) for run in runs)


def measure(reps: int, csv_path: Path, host: str) -> list[Run]:
    """Measure n = reps per pattern, one job at a time."""
    from ray.job_submission import JobSubmissionClient

    sys.path.insert(0, str(ROOT / "scripts"))
    import ops

    from modelport.config import ClientSettings
    from modelport.registry import MlflowRegistry
    from modelport_platform.batch.plan import plan
    from modelport_platform.canary.prometheus import PrometheusClient

    (item,) = plan(
        MlflowRegistry(ClientSettings.from_env().mlflow_url), ops.dataset_store(), [MODEL], BENCH_DAY
    )
    shared = json.loads(kubectl("get", "raycluster", "modelport-shared", "-n", "ray", "-o", "json"))
    for key in ("status",):
        shared.pop(key, None)
    env, stamp = cluster_env(), now().strftime("%Y%m%dT%H%M%SZ")
    client, prometheus = JobSubmissionClient(RAY_DASHBOARD_URL), PrometheusClient(PROMETHEUS_URL)
    runs: list[Run] = []
    for rep in range(1, reps + 1):
        jobs = jobs_for(item, env, stamp, rep)
        wait_idle()
        cold = run_shared(client, jobs["shared"], rep, "shared")
        warm = run_shared(client, jobs["shared-warm"], rep, "shared-warm")
        add_cpu([cold, warm], prometheus)
        wait_idle()
        manifest = rayjob_manifest(
            shared, name=rayjob_name(stamp, rep), job=jobs["rayjob"], image=env["MODELPORT_RAY_IMAGE"]
        )
        rayjob = run_rayjob(manifest, rep)
        add_cpu([rayjob], prometheus)
        kubectl("delete", "rayjob", rayjob.job_id, "-n", "ray", "--ignore-not-found", "--wait=true")
        runs += [cold, warm, rayjob]
        write_csv(csv_path, runs, host)
        print(
            f"bench: rep {rep}: " + ", ".join(json.dumps(r.row(host)) for r in (cold, warm, rayjob)),
            flush=True,
        )
    return runs


def push_last(runs: Sequence[Run]) -> None:
    """modelport_job_*{kind="bench"} of the last rayjob and shared runs (needs MODELPORT_PUSHGATEWAY_URL)."""
    from modelport_platform.metrics.push import push_job_timing

    url = os.environ.get("MODELPORT_PUSHGATEWAY_URL") or None
    for pattern in ("rayjob", "shared"):
        last = [r for r in runs if r.pattern == pattern][-1]
        push_job_timing(
            url,
            model=MODEL,
            pattern=pattern,  # type: ignore[arg-type]
            kind="bench",
            startup_s=(last.running - last.submitted).total_seconds(),
            duration_s=(last.ended - last.running).total_seconds(),
            cpu_s=last.cpu_s,
        )


def drills(host: str, profile: str) -> bool:
    """Each drill once, judged by its verify.py check; one outcome line each in docs/results/drills.md."""
    lines = []
    for drill, model, check in DRILLS:
        ran = sh("bash", f"bench/drills/{drill}.sh", model) == 0
        verdict = subprocess.run(  # noqa: S603 - fixed command
            [
                *COMMANDS["ops"][:3],
                "scripts/verify.py",
                "--profile",
                profile,
                "--model",
                "fashion-classifier",
                "--checks",
                check,
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        print(verdict.stdout, verdict.stderr, sep="", flush=True)
        passed = ran and verdict.returncode == 0
        lines.append(
            f"- {now():%Y-%m-%d}, {host}: `{drill}.sh {model}`: {'PASS' if passed else 'FAIL'} ({check})"
        )
    path = ROOT / "docs/results/drills.md"
    if not path.is_file():
        path.write_text(
            "# Failure drills\n\nOne line per drill per `make bench` run.\n\n", encoding="utf-8", newline="\n"
        )
    with path.open("a", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")
    return all(" PASS " in line for line in lines)


def main(argv: Sequence[str] | None = None) -> int:
    """Measure, run the drills, then regenerate the table."""
    parser = argparse.ArgumentParser(description="FR-13 bench: RayJob vs shared cluster, then the drills")
    parser.add_argument("--reps", type=int, default=5, help="runs per pattern (n)")
    parser.add_argument("--profile", choices=("smoke", "full"), help="default: the running stack's")
    parser.add_argument("--skip-drills", action="store_true")
    parser.add_argument("--dry-run-check", action="store_true", help="print the plan; touch no cluster")
    args = parser.parse_args(argv)
    if args.dry_run_check:
        return dry_run(args.reps)
    import summarize

    host = host_label()
    onboard()
    csv_path = ROOT / f"docs/results/raw/bench-{now():%Y%m%dT%H%M%SZ}.csv"
    runs = measure(args.reps, csv_path, host)
    push_last(runs)
    profile = args.profile or cluster_env().get("MODELPORT_PROFILE", "smoke")
    drills_ok = True if args.skip_drills else drills(host, profile)
    summarize.main([])
    print(f"bench: wrote {csv_path.relative_to(ROOT)}")
    return 0 if drills_ok else 1


if __name__ == "__main__":
    sys.exit(main())
