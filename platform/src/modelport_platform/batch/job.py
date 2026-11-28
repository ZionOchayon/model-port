"""The batch inference job (runs on the shared RayCluster): score one day's input with one pinned version.

`python -m modelport_platform.batch.job --model NAME --version V --input-uri S3URI --output-uri S3URI
--day YYYY-MM-DD --batch-size N --num-cpus C`

The output folder is overwritten as a whole, then `_SUCCESS.json` is written: a rerun never duplicates rows.
"""

import argparse
import json
import sys
import time
from collections.abc import Sequence
from datetime import UTC, date, datetime
from typing import TYPE_CHECKING

import pandas as pd

from modelport.logs import configure_logging, get_logger

if TYPE_CHECKING:
    from pyarrow.fs import FileSystem

_log = get_logger(__name__)
SUCCESS_MARKER = "_SUCCESS.json"


class Scorer:
    """Ray Data callable class: loads the version once per actor and scores pandas batches."""

    def __init__(self, model_uri: str, tracking_uri: str | None = None) -> None:
        import mlflow

        from modelport.pyfunc import load_version

        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)  # actors do not inherit the driver's MLflow state
        self.model = load_version(model_uri).model
        self.name, _, self.version = model_uri.removeprefix("models:/").partition("/")
        self.inputs = [c.name for c in self.model.spec.signature.inputs]

    def __call__(self, batch: pd.DataFrame) -> pd.DataFrame:
        """The batch's columns plus `prediction`, `confidence`, `model` and `version`."""
        scored = self.model.run_inference(batch[self.inputs])
        return batch.reset_index(drop=True).assign(
            prediction=scored["prediction"].to_numpy(),
            confidence=scored["confidence"].to_numpy(),
            model=self.name,
            version=self.version,
        )


def _path(uri: str) -> str:
    return uri.removeprefix("s3://").rstrip("/")


def run_batch(
    *,
    model: str,
    version: str,
    input_uri: str,
    output_uri: str,
    day: date,
    filesystem: "FileSystem",
    tracking_uri: str | None,
    batch_size: int,
    num_cpus: float,
) -> int:
    """Score every input row into `output_uri` (replacing what was there) and return the row count."""
    import ray.data
    from ray.data import SaveMode

    start = time.perf_counter()
    scored = ray.data.read_parquet(_path(input_uri), filesystem=filesystem).map_batches(
        Scorer,
        fn_constructor_kwargs={"model_uri": f"models:/{model}/{version}", "tracking_uri": tracking_uri},
        batch_format="pandas",
        batch_size=batch_size,
        concurrency=(1, 2),
        num_cpus=num_cpus,
    )
    out = _path(output_uri)
    scored.write_parquet(out, filesystem=filesystem, mode=SaveMode.OVERWRITE)
    rows = int(ray.data.read_parquet(out, filesystem=filesystem).count())
    marker = {
        "model": model,
        "version": version,
        "rows": rows,
        "input_uri": input_uri,
        "day": day.isoformat(),
        "finished_at": datetime.now(UTC).isoformat(),
    }
    with filesystem.open_output_stream(f"{out}/{SUCCESS_MARKER}") as stream:
        stream.write(json.dumps(marker).encode())
    _log.info(
        "batch.finished",
        model=model,
        version=version,
        rows=rows,
        seconds=round(time.perf_counter() - start, 1),
    )
    return rows


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """The job's command line."""
    parser = argparse.ArgumentParser(prog="modelport_platform.batch.job")
    parser.add_argument("--model", required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--input-uri", required=True)
    parser.add_argument("--output-uri", required=True)
    parser.add_argument("--day", required=True, type=date.fromisoformat)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-cpus", type=float, default=1.0)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run inside a shared-cluster job with the S3 filesystem and the platform's MLflow."""
    from modelport_platform.s3 import arrow_filesystem
    from modelport_platform.settings import PlatformSettings

    configure_logging(json=True)
    args = parse_args(argv)
    settings = PlatformSettings.from_env()
    run_batch(
        model=args.model,
        version=args.version,
        input_uri=args.input_uri,
        output_uri=args.output_uri,
        day=args.day,
        filesystem=arrow_filesystem(settings),
        tracking_uri=settings.mlflow_uri,
        batch_size=args.batch_size,
        num_cpus=args.num_cpus,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
