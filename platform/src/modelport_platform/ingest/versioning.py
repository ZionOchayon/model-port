"""Turn a valid file into an immutable dataset version, or quarantine an invalid one.

A version is `s3://datasets/<dataset>/v<n>/{train,val}.parquet` plus `manifest.json`, written **last**: a
version exists only once its manifest exists, so a crashed attempt leaves a folder that is simply
overwritten. Versions are keyed by the file's sha256, so the same file never becomes two versions.
"""

import io
import json
import math
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import PurePosixPath
from typing import Final

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from modelport.errors import NotFoundError
from modelport.logs import get_logger
from modelport.naming import dataset_root, dataset_version_uri, quarantine_uri
from modelport_platform.ingest.quality import DatasetContract, QualityReport
from modelport_platform.ports import DatasetStore

VAL_FRACTION: Final = 0.1
_MANIFEST = re.compile(r"/v([1-9][0-9]*)/manifest\.json$")
_log = get_logger(__name__)


@dataclass(frozen=True)
class DatasetVersion:
    """One immutable dataset version."""

    dataset: str
    version: int
    uri: str  # s3://datasets/<dataset>/v<n>
    sha256: str
    rows_train: int
    rows_val: int
    labels: tuple[str, ...]
    created: bool  # False when an identical version already existed


def load_contract(store: DatasetStore, dataset: str) -> DatasetContract:
    """The dataset contract.

    Raises:
        NotFoundError: the dataset has no contract.json.
    """
    return DatasetContract.from_json(json.loads(store.read_bytes(f"{dataset_root(dataset)}/contract.json")))


def _versions(store: DatasetStore, dataset: str) -> list[DatasetVersion]:
    """Every committed version (a folder with a manifest), oldest first."""
    found = []
    for uri in store.list(f"{dataset_root(dataset)}/"):
        if _MANIFEST.search(uri):
            m = json.loads(store.read_bytes(uri))
            found.append(
                DatasetVersion(
                    dataset=dataset,
                    version=int(m["version"]),
                    uri=dataset_version_uri(dataset, int(m["version"])),
                    sha256=m["sha256"],
                    rows_train=int(m["rows_train"]),
                    rows_val=int(m["rows_val"]),
                    labels=tuple(m["labels"]),
                    created=False,
                )
            )
    return sorted(found, key=lambda v: v.version)


def find_version_by_sha(store: DatasetStore, dataset: str, sha256: str) -> DatasetVersion | None:
    """The version made from a file with this sha256, if any."""
    return next((v for v in _versions(store, dataset) if v.sha256 == sha256), None)


def latest_version(store: DatasetStore, dataset: str) -> DatasetVersion | None:
    """The newest committed version, if any."""
    versions = _versions(store, dataset)
    return versions[-1] if versions else None


def split_train_val(
    table: pa.Table, sha256: str, val_fraction: float = VAL_FRACTION
) -> tuple[pa.Table, pa.Table]:
    """Split seeded by the file hash: the first ceil(rows x fraction) of a permutation are val."""
    order = np.random.default_rng(int(sha256[:16], 16)).permutation(table.num_rows)
    n_val = math.ceil(table.num_rows * val_fraction)
    return table.take(order[n_val:]), table.take(order[:n_val])


def _parquet(table: pa.Table) -> bytes:
    buffer = io.BytesIO()
    pq.write_table(table, buffer)
    return buffer.getvalue()


def write_version(
    store: DatasetStore, dataset: str, table: pa.Table, sha256: str, source_uri: str, report: QualityReport
) -> DatasetVersion:
    """Write the next version (or return the existing one for the same sha256); manifest last."""
    versions = _versions(store, dataset)
    existing = next((v for v in versions if v.sha256 == sha256), None)
    if existing is not None:
        _log.info("ingest.versioned", dataset=dataset, version=existing.version, created=False)
        return existing
    number = versions[-1].version + 1 if versions else 1
    uri = dataset_version_uri(dataset, number)
    train, val = split_train_val(table, sha256)
    labels = tuple(sorted(report.class_counts))
    store.write_bytes(f"{uri}/train.parquet", _parquet(train))
    store.write_bytes(f"{uri}/val.parquet", _parquet(val))
    manifest = {
        "dataset": dataset,
        "version": number,
        "sha256": sha256,
        "source_uri": source_uri,
        "rows_train": train.num_rows,
        "rows_val": val.num_rows,
        "labels": list(labels),
        "class_counts": dict(report.class_counts),
        "created_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "quality": {"passed": report.passed, "checks": report.to_json()["checks"]},
    }
    store.write_bytes(
        f"{uri}/manifest.json", json.dumps(manifest, indent=2).encode(), content_type="application/json"
    )
    _log.info("ingest.versioned", dataset=dataset, version=number, created=True)
    return DatasetVersion(dataset, number, uri, sha256, train.num_rows, val.num_rows, labels, created=True)


def quarantine(store: DatasetStore, dataset: str, source_uri: str, sha256: str, report: QualityReport) -> str:
    """Copy the file to s3://quarantine/<dataset>/<sha256>/ with report.json; return that folder."""
    target = quarantine_uri(dataset, sha256)
    try:
        store.copy(source_uri, f"{target}{PurePosixPath(source_uri).name}")
    except NotFoundError:
        _log.warning("ingest.quarantine_source_missing", source_uri=source_uri)
    store.write_bytes(
        f"{target}report.json",
        json.dumps(report.to_json(), indent=2).encode(),
        content_type="application/json",
    )
    _log.info(
        "ingest.quarantined",
        dataset=dataset,
        sha256=sha256,
        failed=[c.name for c in report.checks if not c.passed],
    )
    return target
