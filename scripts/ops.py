"""Platform-engineer operations against a running stack (used by CI and `make demo`).

Usage: uv run python scripts/ops.py <command> ... ; exit 0 success, 1 failure, 124 timeout.
Reads .modelport/credentials.env and talks to the port-forwarded services (`make ui`).

Commands:
  wait-dataset <dataset> <version> [--timeout S]   s3://datasets/<ds>/v<version>/manifest.json exists
  wait-quarantine <dataset> [--timeout S]          some report.json exists under s3://quarantine/<ds>/
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections.abc import Callable
from pathlib import Path

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
    args = parser.parse_args(argv)
    code: int = args.run(args)
    return code


if __name__ == "__main__":
    sys.exit(main())
