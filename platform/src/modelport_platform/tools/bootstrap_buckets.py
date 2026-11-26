"""Create every modelport S3 bucket idempotently (Job `minio-buckets`, run by `make up`).

Usage: python -m modelport_platform.tools.bootstrap_buckets
Credentials and endpoint come from the environment (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY,
AWS_ENDPOINT_URL, AWS_REGION), which the Job gets from the platform ConfigMap and Secret.
"""

from __future__ import annotations

import sys
from typing import Any

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

from modelport.naming import Bucket

type S3Client = Any  # boto3 ships no type stubs

_ALREADY_THERE = {"BucketAlreadyOwnedByYou", "BucketAlreadyExists"}


def s3_client() -> S3Client:
    """S3 client with path-style addressing and standard retries (MinIO in-cluster)."""
    config = Config(s3={"addressing_style": "path"}, retries={"max_attempts": 3, "mode": "standard"})
    return boto3.client("s3", config=config)


def bootstrap(client: S3Client) -> list[str]:
    """Create each bucket of `Bucket` that does not exist yet; return the names created."""
    created: list[str] = []
    for bucket in Bucket:
        try:
            client.create_bucket(Bucket=bucket.value)
        except ClientError as err:
            if err.response.get("Error", {}).get("Code") not in _ALREADY_THERE:
                raise
        else:
            created.append(bucket.value)
    return created


def missing_buckets(client: S3Client) -> list[str]:
    """Names of buckets that `head_bucket` cannot see."""
    missing: list[str] = []
    for bucket in Bucket:
        try:
            client.head_bucket(Bucket=bucket.value)
        except ClientError:
            missing.append(bucket.value)
    return missing


def main() -> int:
    """Create the buckets, confirm each exists, and return the process exit code."""
    client = s3_client()
    created = bootstrap(client)
    missing = missing_buckets(client)
    print(f"bootstrap_buckets: created {created or 'none'}; missing {missing or 'none'}")
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
