"""Platform-engineer operations against a running stack (used by CI and `make demo`).

Usage: uv run python scripts/ops.py <command> ... ; exit 0 success, 1 failure, 124 timeout.
Reads .modelport/credentials.env and talks to the port-forwarded services (`make ui`).

Commands:
  wait-dataset <dataset> <version> [--timeout S]   s3://datasets/<ds>/v<version>/manifest.json exists
  wait-quarantine <dataset> [--timeout S]          some report.json exists under s3://quarantine/<ds>/
  wait-dag-run <dag_id> <run_id> [--state success] [--timeout S]   the run reaches the state (fails fast
                                                   on the other terminal state)
  wait-alias <model> <alias> [--timeout S]         the alias exists in MLflow
  trigger <dag_id> <run_id> [--conf JSON] [--wait] [--timeout S]   start a DAG run (idempotent)
  snapshot-predictions --day D --out FILE          row counts and row hashes of s3://predictions/*/<D>/
  wait-checkpoint <model> <rayjob> [--timeout S]   Ray Train registered a checkpoint of that run
  wait-metric <promql> [--timeout S]               the aggregating PromQL query has a value in Prometheus
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
POLL_S = 10


def credentials() -> dict[str, str]:
    """Environment first, then .modelport/credentials.env."""
    values = dict(os.environ)
    path = ROOT / ".modelport" / "credentials.env"
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                key, _, value = line.partition("=")
                values.setdefault(key.strip(), value.strip())
    return values


def dataset_store():  # noqa: ANN201 - S3DatasetStore; imported lazily so --help works anywhere
    """The S3 store through the port-forward (MODELPORT_S3_* from credentials.env)."""
    import boto3
    from botocore.config import Config

    from modelport_platform.s3 import S3DatasetStore

    env = credentials()
    client = boto3.client(
        "s3",
        endpoint_url=env.get("MODELPORT_S3_ENDPOINT", "http://localhost:9000"),
        aws_access_key_id=env["MODELPORT_S3_ACCESS_KEY"],
        aws_secret_access_key=env["MODELPORT_S3_SECRET_KEY"],
        region_name="us-east-1",
        config=Config(s3={"addressing_style": "path"}),
    )
    return S3DatasetStore(client)


def poll(what: str, done: Callable[[], bool], timeout_s: float) -> int:
    """Call `done` every 10 s until it is true (0) or the timeout passes (124)."""
    deadline = time.monotonic() + timeout_s
    while True:
        try:
            if done():
                print(f"ops: {what}: ok")
                return 0
        except Exception as err:  # transient (port-forward restarting, object not yet there)
            print(f"ops: {what}: waiting ({type(err).__name__}: {err})")
        if time.monotonic() >= deadline:
            print(f"ops: {what}: timed out after {timeout_s:.0f} s", file=sys.stderr)
            return 124
        time.sleep(POLL_S)


def wait_dataset(args: argparse.Namespace) -> int:
    """Wait for a committed dataset version."""
    from modelport.naming import dataset_version_uri

    uri = f"{dataset_version_uri(args.dataset, args.version)}/manifest.json"
    store = dataset_store()
    return poll(f"wait-dataset {uri}", lambda: store.exists(uri), args.timeout)


def wait_quarantine(args: argparse.Namespace) -> int:
    """Wait for any quarantine report of a dataset."""
    from modelport.naming import Bucket

    store = dataset_store()
    prefix = f"s3://{Bucket.QUARANTINE}/{args.dataset}/"
    return poll(
        f"wait-quarantine {prefix}",
        lambda: any(u.endswith("/report.json") for u in store.list(prefix)),
        args.timeout,
    )


def airflow():  # noqa: ANN201 - AirflowClient
    """The Airflow REST client through the port-forward."""
    from modelport.clients.airflow import AirflowClient

    env = credentials()
    return AirflowClient(
        env.get("MODELPORT_AIRFLOW_URL", "http://localhost:8080"),
        env.get("MODELPORT_AIRFLOW_USERNAME", "modelport"),
        env["MODELPORT_AIRFLOW_PASSWORD"],
    )


def run_state_done(state: str | None, wanted: str) -> bool:
    """True when the run reached `wanted`; raises when it ended in the other terminal state."""
    if state == wanted:
        return True
    if state in {"success", "failed"}:
        raise SystemExit(f"ops: run ended {state}, expected {wanted}")
    return False


def wait_dag_run(args: argparse.Namespace) -> int:
    """Wait for a DAG run to reach a state."""
    client = airflow()
    if args.if_exists and client.get_run(args.dag_id, args.run_id) is None:
        print(f"ops: wait-dag-run {args.dag_id}/{args.run_id}: no such run, nothing to wait for")
        return 0

    def done() -> bool:
        info = client.get_run(args.dag_id, args.run_id)
        return run_state_done(info.state if info else None, args.state)

    try:
        return poll(f"wait-dag-run {args.dag_id}/{args.run_id} {args.state}", done, args.timeout)
    except SystemExit as err:
        print(err, file=sys.stderr)
        return 1


def wait_alias(args: argparse.Namespace) -> int:
    """Wait until a registered model has the alias."""
    from modelport.naming import Alias
    from modelport.registry import MlflowRegistry

    registry = MlflowRegistry(credentials().get("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    alias = Alias(args.alias)
    return poll(
        f"wait-alias {args.model} {alias}",
        lambda: registry.resolve(args.model, alias) is not None,
        args.timeout,
    )


def trigger(args: argparse.Namespace) -> int:
    """Start a DAG run (an existing run with the same ID is reused) and optionally wait for it."""
    from modelport.errors import ModelportError

    client = airflow()
    try:
        ref = client.trigger(args.dag_id, args.run_id, json.loads(args.conf))
        how = "existing" if ref.existed else "started"
        print(f"ops: trigger {args.dag_id}/{args.run_id}: {how} {ref.url}")
        if args.wait:
            client.wait(args.dag_id, args.run_id, timeout_s=args.timeout)
            print(f"ops: {args.dag_id}/{args.run_id}: success")
    except ModelportError as err:
        print(f"ops: {err.what}\n{err.fix}", file=sys.stderr)
        return 124 if type(err).__name__ == "WaitTimeoutError" else 1
    return 0


def row_hashes(frame: Any) -> list[str]:  # noqa: ANN401 - pandas.DataFrame
    """Sorted sha256 of every input row + prediction + confidence rounded to 6 places (order-free)."""
    inputs = [c for c in frame.columns if c not in {"prediction", "confidence", "model", "version"}]
    hashes = []
    for row in frame.itertuples(index=False):
        values = row._asdict()
        payload = json.dumps([str(values[c]) for c in inputs], sort_keys=True)
        row_id = hashlib.sha256(payload.encode()).hexdigest()
        line = f"{row_id}|{values['prediction']}|{round(float(values['confidence']), 6)}"
        hashes.append(hashlib.sha256(line.encode()).hexdigest())
    return sorted(hashes)


def snapshot_predictions(args: argparse.Namespace) -> int:
    """Write {model: {rows, hashes}} for every model's predictions of a day."""
    import pandas as pd

    store = dataset_store()
    snapshot: dict[str, Any] = {}
    for uri in store.list("s3://predictions/"):
        parts = uri.removeprefix("s3://predictions/").split("/")
        if len(parts) < 3 or parts[1] != args.day or not uri.endswith(".parquet"):
            continue
        frame = pd.read_parquet(io.BytesIO(store.read_bytes(uri)))
        entry = snapshot.setdefault(parts[0], {"rows": 0, "hashes": []})
        entry["rows"] += len(frame)
        entry["hashes"] = sorted(entry["hashes"] + row_hashes(frame))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(snapshot, indent=2, sort_keys=True), encoding="utf-8")
    counts = ", ".join(f"{m} {e['rows']} rows" for m, e in snapshot.items())
    print(f"ops: snapshot-predictions {args.day}: {counts}")
    return 0 if snapshot else 1


def snapshot_has_checkpoint(text: str) -> bool:
    """Whether a Ray Train checkpoint-manager snapshot lists at least one checkpoint."""
    try:
        state = json.loads(text)
    except ValueError:
        return False
    return "checkpoint_" in json.dumps(state.get("checkpoint_results", state))


def wait_checkpoint(args: argparse.Namespace) -> int:
    """Wait until Ray Train has registered a checkpoint for the RayJob's run."""
    from modelport.naming import checkpoint_root

    uri = f"{checkpoint_root(args.model)}/{args.rayjob}/checkpoint_manager_snapshot.json"
    store = dataset_store()
    return poll(
        f"wait-checkpoint {uri}",
        lambda: store.exists(uri) and snapshot_has_checkpoint(store.read_bytes(uri).decode("utf-8")),
        args.timeout,
    )


def wait_metric(args: argparse.Namespace) -> int:
    """Wait until an aggregating PromQL query returns a value."""
    from modelport_platform.canary.prometheus import PrometheusClient

    client = PrometheusClient(credentials().get("MODELPORT_PROMETHEUS_URL", "http://localhost:9090"))
    return poll(f"wait-metric {args.promql}", lambda: client.scalar(args.promql) is not None, args.timeout)


def main(argv: list[str] | None = None) -> int:
    """Dispatch a subcommand."""
    parser = argparse.ArgumentParser(description="modelport platform operations")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("wait-dataset", help="wait for s3://datasets/<ds>/v<n>/manifest.json")
    p.add_argument("dataset")
    p.add_argument("version", type=int)
    p.add_argument("--timeout", type=float, default=900)
    p.set_defaults(run=wait_dataset)
    p = sub.add_parser("wait-quarantine", help="wait for a quarantine report.json")
    p.add_argument("dataset")
    p.add_argument("--timeout", type=float, default=900)
    p.set_defaults(run=wait_quarantine)
    p = sub.add_parser("wait-dag-run", help="wait for an Airflow DAG run to reach a state")
    p.add_argument("dag_id")
    p.add_argument("run_id")
    p.add_argument("--state", choices=("success", "failed"), default="success")
    p.add_argument("--if-exists", action="store_true", help="succeed at once when the run does not exist")
    p.add_argument("--timeout", type=float, default=3600)
    p.set_defaults(run=wait_dag_run)
    p = sub.add_parser("wait-alias", help="wait for an MLflow alias of a model")
    p.add_argument("model")
    p.add_argument("alias", choices=("champion", "previous", "candidate"))
    p.add_argument("--timeout", type=float, default=2400)
    p.set_defaults(run=wait_alias)
    p = sub.add_parser("trigger", help="start a DAG run")
    p.add_argument("dag_id")
    p.add_argument("run_id")
    p.add_argument("--conf", default="{}")
    p.add_argument("--wait", action="store_true")
    p.add_argument("--timeout", type=float, default=3600)
    p.set_defaults(run=trigger)
    p = sub.add_parser("snapshot-predictions", help="row counts and hashes of a day's predictions")
    p.add_argument("--day", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(run=snapshot_predictions)
    p = sub.add_parser("wait-checkpoint", help="wait for a registered Ray Train checkpoint of a RayJob")
    p.add_argument("model")
    p.add_argument("rayjob")
    p.add_argument("--timeout", type=float, default=1800)
    p.set_defaults(run=wait_checkpoint)
    p = sub.add_parser("wait-metric", help="wait for a PromQL query to return a value")
    p.add_argument("promql")
    p.add_argument("--timeout", type=float, default=300)
    p.set_defaults(run=wait_metric)
    args = parser.parse_args(argv)
    code: int = args.run(args)
    return code


if __name__ == "__main__":
    sys.exit(main())
