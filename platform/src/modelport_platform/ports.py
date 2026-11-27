"""The ports (Protocols) the platform core depends on; adapters live in s3, kube and canary.prometheus."""

from collections.abc import Mapping
from datetime import datetime
from typing import Any, Protocol


class DatasetStore(Protocol):
    """Objects addressed by `s3://bucket/key` URIs."""

    def read_bytes(self, uri: str) -> bytes:
        """The object's bytes; NotFoundError if it does not exist."""

    def write_bytes(self, uri: str, data: bytes, *, content_type: str = "application/octet-stream") -> None:
        """Create or replace the object."""

    def exists(self, uri: str) -> bool:
        """Whether the object exists."""

    def list(self, prefix_uri: str) -> list[str]:
        """URIs of every object under the prefix, sorted."""

    def copy(self, src_uri: str, dst_uri: str) -> None:
        """Copy one object."""

    def delete(self, uri: str) -> None:
        """Delete one object (missing is fine)."""


class MetricsQuery(Protocol):
    """PromQL queries."""

    def scalar(self, promql: str, *, at: datetime | None = None) -> float | None:
        """One aggregated value, or None for an empty result."""

    def vector(self, promql: str, *, at: datetime | None = None) -> list[tuple[Mapping[str, str], float]]:
        """(labels, value) per series."""


class KubeApi(Protocol):
    """KubeRay custom objects in one namespace."""

    def ensure(self, plural: str, body: Mapping[str, Any]) -> tuple[dict[str, Any], bool]:
        """Create the object, or return the existing one; the bool says whether it was created."""

    def get(self, plural: str, name: str) -> dict[str, Any] | None:
        """The object, or None."""

    def merge_patch(self, plural: str, name: str, patch: Mapping[str, Any]) -> dict[str, Any]:
        """Apply a JSON merge patch and return the result."""

    def delete(self, plural: str, name: str) -> None:
        """Delete the object (missing is fine)."""
