"""S3 (MinIO) adapters: boto3 and pyarrow clients from settings, and the DatasetStore adapter."""

from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

from modelport.errors import NotFoundError
from modelport.logs import get_logger
from modelport_platform.settings import PlatformSettings

if TYPE_CHECKING:
    from pyarrow.fs import S3FileSystem

type S3Client = Any  # boto3 ships no type stubs
_MISSING = frozenset({"NoSuchKey", "404", "NotFound"})
_log = get_logger(__name__)


def boto3_client(settings: PlatformSettings) -> S3Client:
    """boto3 S3 client for MinIO: path-style addressing, 3 attempts."""
    return boto3.client(
        "s3",
        endpoint_url=settings.s3_endpoint,
        aws_access_key_id=settings.s3_access_key,
        aws_secret_access_key=settings.s3_secret_key,
        region_name=settings.s3_region,
        config=Config(s3={"addressing_style": "path"}, retries={"max_attempts": 3, "mode": "standard"}),
    )


def arrow_filesystem(settings: PlatformSettings) -> "S3FileSystem":
    """The pyarrow S3 filesystem for Ray Data and Ray Train storage."""
    from pyarrow.fs import S3FileSystem

    endpoint = urlparse(settings.s3_endpoint)
    return S3FileSystem(
        access_key=settings.s3_access_key,
        secret_key=settings.s3_secret_key,
        endpoint_override=endpoint.netloc,
        scheme=endpoint.scheme or "http",
        region=settings.s3_region,
    )


def split_uri(uri: str) -> tuple[str, str]:
    """`s3://bucket/key` -> ("bucket", "key")."""
    if not uri.startswith("s3://"):
        raise ValueError(f"not an s3:// URI: {uri}")
    bucket, _, key = uri.removeprefix("s3://").partition("/")
    return bucket, key


class S3DatasetStore:
    """`DatasetStore` over a boto3 client."""

    def __init__(self, client: S3Client) -> None:
        self._s3 = client

    def read_bytes(self, uri: str) -> bytes:
        """The object's bytes.

        Raises:
            NotFoundError: the object does not exist.
        """
        bucket, key = split_uri(uri)
        try:
            data: bytes = self._s3.get_object(Bucket=bucket, Key=key)["Body"].read()
        except ClientError as err:
            if err.response.get("Error", {}).get("Code") in _MISSING:
                raise NotFoundError(
                    what=f"{uri} does not exist", fix="Check the dataset name and version."
                ) from err
            raise
        return data

    def write_bytes(self, uri: str, data: bytes, *, content_type: str = "application/octet-stream") -> None:
        """Create or replace the object."""
        bucket, key = split_uri(uri)
        self._s3.put_object(Bucket=bucket, Key=key, Body=data, ContentType=content_type)
        _log.debug("s3.write", uri=uri, bytes=len(data))

    def exists(self, uri: str) -> bool:
        """Whether the object exists."""
        bucket, key = split_uri(uri)
        try:
            self._s3.head_object(Bucket=bucket, Key=key)
        except ClientError as err:
            if err.response.get("Error", {}).get("Code") in _MISSING:
                return False
            raise
        return True

    def list(self, prefix_uri: str) -> list[str]:
        """URIs of every object under the prefix, sorted."""
        bucket, prefix = split_uri(prefix_uri)
        keys = [
            item["Key"]
            for page in self._s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix)
            for item in page.get("Contents", [])
        ]
        return [f"s3://{bucket}/{key}" for key in sorted(keys)]

    def copy(self, src_uri: str, dst_uri: str) -> None:
        """Copy one object (server side)."""
        src_bucket, src_key = split_uri(src_uri)
        dst_bucket, dst_key = split_uri(dst_uri)
        self._s3.copy_object(
            Bucket=dst_bucket, Key=dst_key, CopySource={"Bucket": src_bucket, "Key": src_key}
        )

    def delete(self, uri: str) -> None:
        """Delete one object; deleting a missing object is fine (S3 semantics)."""
        bucket, key = split_uri(uri)
        self._s3.delete_object(Bucket=bucket, Key=key)
