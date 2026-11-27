"""Dataset versions for training: which version to use, and reading its splits into Ray Data."""

from typing import TYPE_CHECKING

from modelport.errors import SpecError
from modelport.spec import ModelSpec
from modelport_platform.ingest.versioning import DatasetVersion, latest_version, version_by_number
from modelport_platform.ports import DatasetStore
from modelport_platform.s3 import arrow_filesystem
from modelport_platform.settings import PlatformSettings

if TYPE_CHECKING:
    import ray.data


def resolve_dataset_version(store: DatasetStore, spec: ModelSpec, override: int | None) -> DatasetVersion:
    """`--dataset-version`, else the version pinned in model.yaml, else the latest.

    Raises:
        SpecError: the dataset has no versions, or the requested version does not exist.
    """
    dataset = spec.data.dataset_name
    wanted = override or spec.data.pinned_version
    version = version_by_number(store, dataset, wanted) if wanted else latest_version(store, dataset)
    if version is None:
        if wanted:
            raise SpecError(
                what=f"Dataset '{dataset}' has no version v{wanted}",
                fix=f"Use an existing version, or upload data to s3://incoming/{dataset}/.",
            )
        raise SpecError(
            what=f"Dataset '{dataset}' has no versions yet",
            fix=f"Upload a file to s3://incoming/{dataset}/ (examples/datasets/prepare.py --upload).",
        )
    return version


def load_split(settings: PlatformSettings, version: DatasetVersion, split: str) -> "ray.data.Dataset":
    """Read `<version>/<split>.parquet` with the S3 filesystem (paths are `bucket/key` with a filesystem)."""
    import ray.data

    path = f"{version.uri.removeprefix('s3://')}/{split}.parquet"
    return ray.data.read_parquet(path, filesystem=arrow_filesystem(settings))
