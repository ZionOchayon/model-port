"""The single source of every name modelport uses: buckets, aliases, tags, paths and IDs."""

from enum import StrEnum


class Bucket(StrEnum):
    """S3 buckets created by `make up`."""

    INCOMING = "incoming"
    QUARANTINE = "quarantine"
    DATASETS = "datasets"
    PREDICTIONS = "predictions"
    MLFLOW = "mlflow"
    CHECKPOINTS = "checkpoints"
    INFERENCE_LOGS = "inference-logs"
    BATCH_INPUT = "batch-input"
