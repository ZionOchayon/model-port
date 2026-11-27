"""ingest_dataset: watch s3://incoming/<dataset>/, check each file, then version it or quarantine it (FR-8).

The S3 sensor is deferrable, so waiting holds no worker slot. A new version emits an Asset event through the
alias `modelport-datasets`, which schedules `retrain_on_dataset`.
"""

import hashlib
import io
from typing import Any, Final

from airflow.providers.amazon.aws.sensors.s3 import S3KeySensor
from airflow.sdk import Asset, AssetAlias, dag, task

from modelport.errors import NotFoundError, SpecError
from modelport.naming import Bucket, asset_uri, incoming_prefix
from modelport_platform import dagkit
from modelport_platform.ingest.quality import QualityCheck, QualityReport, check_table
from modelport_platform.ingest.versioning import load_contract, quarantine, write_version
from modelport_platform.ports import DatasetStore

DATASETS_ALIAS: Final = "modelport-datasets"
MAX_FILES_PER_RUN: Final = 10
UNKNOWN_DATASET: Final = "unknown-dataset"  # quarantine folder for keys outside any valid dataset name


def list_incoming(store: DatasetStore) -> list[dict[str, str]]:
    """Up to 10 `incoming/<dataset>/<file>.parquet` keys, in name order."""
    items = []
    for uri in store.list(f"s3://{Bucket.INCOMING}/"):
        parts = uri.removeprefix(f"s3://{Bucket.INCOMING}/").split("/")
        if len(parts) == 2 and parts[1].endswith(".parquet"):
            items.append({"dataset": parts[0], "uri": uri})
    return items[:MAX_FILES_PER_RUN]


def _no_contract(dataset: str, uri: str, sha256: str) -> QualityReport:
    detail = f"no contract for dataset '{dataset}' (s3://datasets/{dataset}/contract.json)"
    return QualityReport(dataset, uri, sha256, 0, {}, (QualityCheck("schema", False, detail),))


def process_file(item: dict[str, str], store: DatasetStore, outlet_events: Any) -> dict[str, Any]:  # noqa: ANN401
    """Check one incoming file; write a version or quarantine it; always remove it from incoming."""
    import pyarrow.parquet as pq

    dataset, uri = item["dataset"], item["uri"]
    data = store.read_bytes(uri)
    sha256 = hashlib.sha256(data).hexdigest()
    valid = _valid(dataset)
    contract = None
    if valid:
        try:
            contract = load_contract(store, dataset)
        except NotFoundError:
            contract = None
    if contract is None:
        target = quarantine(
            store, dataset if valid else UNKNOWN_DATASET, uri, sha256, _no_contract(dataset, uri, sha256)
        )
        store.delete(uri)
        return {"dataset": dataset, "quarantined": target}
    try:
        table, error = pq.read_table(io.BytesIO(data)), None
    except Exception as err:  # any parse failure means the file is unreadable
        table, error = None, f"{type(err).__name__}: {err}"
    report = check_table(table, error, contract, source_uri=uri, sha256=sha256)
    if not report.passed:
        target = quarantine(store, dataset, uri, sha256, report)
        store.delete(uri)
        return {"dataset": dataset, "quarantined": target}
    assert table is not None
    version = write_version(store, dataset, table, sha256, uri, report)
    store.delete(uri)
    if version.created:
        outlet_events[AssetAlias(DATASETS_ALIAS)].add(
            Asset(asset_uri(dataset)),
            extra={"dataset": dataset, "version": version.version, "sha256": sha256},
        )
    return {"dataset": dataset, "version": version.version, "created": version.created}


def _valid(dataset: str) -> bool:
    try:
        incoming_prefix(dataset)
    except SpecError:
        return False
    return True


@dag(
    dag_id="ingest_dataset",
    schedule="@continuous",
    max_active_runs=1,
    catchup=False,
    default_args=dagkit.DEFAULT_ARGS,
    tags=[*dagkit.TAGS, "ingest"],
    doc_md=__doc__,
)
def ingest_dataset() -> None:
    """Wait for files in s3://incoming, then process each one."""
    wait_for_file = S3KeySensor(
        task_id="wait_for_file",
        bucket_key=f"s3://{Bucket.INCOMING}/*/*.parquet",
        wildcard_match=True,
        aws_conn_id="minio_s3",
        deferrable=True,
        poke_interval=30,
        timeout=86400,
        soft_fail=True,
    )

    @task
    def list_files() -> list[dict[str, str]]:
        return list_incoming(dagkit.store())

    @task(outlets=[AssetAlias(DATASETS_ALIAS)])
    def process(item: dict[str, str], outlet_events: Any = None) -> dict[str, Any]:  # noqa: ANN401
        return process_file(item, dagkit.store(), outlet_events)

    files = list_files()
    wait_for_file >> files
    process.expand(item=files)


ingest_dataset()
