"""The evaluation job (runs on the shared RayCluster): measure one version on one validation set.

`python -m modelport_platform.gate.evaluate --model NAME --version V --val-uri S3URI --report-run-id RUN_ID`

It measures and never decides: the DAG compares the reports. Exit codes: 0 ok, 1 unexpected,
3 the registered signature rejects the validation data, 4 the label column is missing.
"""

import argparse
import statistics
import sys
import time
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from typing import Any, Final, Protocol

import pandas as pd

from modelport.logs import configure_logging, get_logger
from modelport.spec import ModelSpec
from modelport_platform.gate.core import EvalReport
from modelport_platform.gate.metrics import score

_log = get_logger(__name__)
PREDICT_BATCH: Final = 1024


class Inference(Protocol):
    """A loaded BaseModel (only `run_inference` is used)."""

    def run_inference(self, batch: pd.DataFrame) -> pd.DataFrame:
        """DataFrame[prediction, confidence]."""


class ReportStore(Protocol):
    """Where the report goes (the MLflow run the DAG created)."""

    def log_json(self, run_id: str, artifact_path: str, payload: Any) -> None:  # noqa: ANN401
        """Write a JSON artifact."""

    def log_metrics(self, run_id: str, metrics: Any, step: int) -> None:  # noqa: ANN401
        """Log metrics."""


def predict_batches(
    model: Inference, frame: pd.DataFrame, *, batch_size: int = PREDICT_BATCH
) -> pd.DataFrame:
    """`run_inference` over `frame` in batches (bounded memory), one result in row order."""
    parts = [model.run_inference(frame.iloc[i : i + batch_size]) for i in range(0, len(frame), batch_size)]
    return pd.concat(parts, ignore_index=True)


def measure_latency(
    model: Inference,
    frame: pd.DataFrame,
    *,
    batch_size: int,
    repeats: int,
    clock: Callable[[], float] = time.perf_counter,
) -> float:
    """Median milliseconds of `run_inference` on the first `batch_size` rows, after one warm-up call."""
    batch = frame.head(batch_size)
    model.run_inference(batch)
    timings = []
    for _ in range(repeats):
        start = clock()
        model.run_inference(batch)
        timings.append((clock() - start) * 1000)
    return float(statistics.median(timings))


def evaluate(
    model: Inference, val: pd.DataFrame, spec: ModelSpec, *, version: str, val_uri: str
) -> EvalReport:
    """Score and time one loaded version on `val` (which holds the label column)."""
    predictions = predict_batches(model, val)
    y_true = val[spec.data.label_column].astype(str).tolist()
    metric, per_class = score(spec.evaluation.metric, y_true, predictions["prediction"].tolist(), spec.labels)
    latency = measure_latency(
        model, val, batch_size=spec.evaluation.latency_batch_size, repeats=spec.evaluation.latency_repeats
    )
    return EvalReport(
        model=spec.name,
        version=version,
        val_uri=val_uri,
        n_samples=len(val),
        metric_name=spec.evaluation.metric,
        metric=metric,
        per_class=per_class,
        latency_ms_median=latency,
        latency_batch_size=spec.evaluation.latency_batch_size,
        created_at=datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """The job's arguments."""
    parser = argparse.ArgumentParser(prog="modelport_platform.gate.evaluate")
    parser.add_argument("--model", required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--val-uri", required=True)
    parser.add_argument("--report-run-id", required=True)
    return parser.parse_args(argv)


def _fail(code: int, what: str, fix: str) -> int:
    _log.error("eval.failed", what=what, fix=fix)
    print(f"Error: {what}\nHow to fix: {fix}", file=sys.stderr)
    return code


def run(args: argparse.Namespace, *, registry: ReportStore, read_val: Callable[[str], pd.DataFrame]) -> int:
    """Load the version, check it accepts the data, evaluate and log the report; return the exit code."""
    import mlflow

    from modelport.pyfunc import load_version

    model_uri = f"models:/{args.model}/{args.version}"
    loaded = load_version(model_uri)
    spec, val = loaded.spec, read_val(args.val_uri)
    label = spec.data.label_column
    if label not in val.columns:
        return _fail(
            4, f"{args.val_uri} has no label column '{label}'", "Fix data.label_column in model.yaml."
        )
    features = val[[c.name for c in spec.signature.inputs]]
    try:
        mlflow.pyfunc.load_model(model_uri).predict(features.head(5))
    except Exception as err:  # MLflow's schema enforcement raises several types
        return _fail(
            3,
            f"The registered signature of {args.model} v{args.version} rejects the validation data: {err}",
            "Make signature.inputs in model.yaml match the dataset's columns and types.",
        )
    report = evaluate(loaded.model, val, spec, version=args.version, val_uri=args.val_uri)
    registry.log_json(args.report_run_id, "gate/report.json", report.to_json())
    registry.log_metrics(
        args.report_run_id,
        {"eval_metric": report.metric, "eval_latency_ms_median": report.latency_ms_median},
        step=0,
    )
    _log.info(
        "eval.finished",
        model=args.model,
        version=args.version,
        metric=report.metric,
        latency_ms_median=report.latency_ms_median,
        n_samples=report.n_samples,
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run inside a shared-cluster job with the real adapters."""
    import mlflow

    from modelport.registry import MlflowRegistry
    from modelport_platform.s3 import arrow_filesystem
    from modelport_platform.settings import PlatformSettings

    configure_logging(json=True)
    settings = PlatformSettings.from_env()
    mlflow.set_tracking_uri(settings.mlflow_uri)

    def read_val(uri: str) -> pd.DataFrame:
        return pd.read_parquet(uri.removeprefix("s3://"), filesystem=arrow_filesystem(settings))

    return run(parse_args(argv), registry=MlflowRegistry(settings.mlflow_uri), read_val=read_val)


if __name__ == "__main__":
    sys.exit(main())
