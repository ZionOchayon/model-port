"""Create every modelport S3 bucket idempotently (Job `minio-buckets`, run by `make up`).

Usage: python -m modelport_platform.tools.bootstrap_buckets
Endpoint and credentials come from the environment (platform settings), which the Job gets from the
platform ConfigMap and the S3 Secret.
"""

from __future__ import annotations

import sys

from botocore.exceptions import ClientError

from modelport.naming import Bucket
from modelport_platform.s3 import S3Client, boto3_client
from modelport_platform.settings import PlatformSettings

_ALREADY_THERE = {"BucketAlreadyOwnedByYou", "BucketAlreadyExists"}


def s3_client() -> S3Client:
    """The platform S3 client, from the Job's environment (platform ConfigMap + S3 Secret)."""
    return boto3_client(PlatformSettings.from_env())


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
