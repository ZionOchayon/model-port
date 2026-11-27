"""Inference log: sampled request rows buffered in a Serve replica and written to S3 for daily drift.

File: `s3://inference-logs/<model>/<YYYY-MM-DD>/<variant>-v<version>-<replica>-<epoch ms>.parquet`, columns
`ts, request_id, model, version, variant, prediction, confidence, stat_<feature>...`. A failed write is
logged and its rows dropped: the log is evidence for drift, never a reason to fail a request.
"""

import io
import random
import time
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from modelport.logs import get_logger
from modelport.naming import inference_log_prefix
from modelport_platform.ports import DatasetStore

_log = get_logger(__name__)


class InferenceLogBuffer:
    """Rows kept in memory and flushed every `flush_rows` rows or `flush_seconds`, whichever comes first."""

    def __init__(
        self,
        *,
        model: str,
        version: str,
        variant: str,
        store: DatasetStore,
        flush_rows: int = 500,
        flush_seconds: float = 60.0,
        sample_rate: float = 1.0,
        replica: str = "local",
        clock: Callable[[], float] = time.time,
        rng: Callable[[], float] = random.random,
    ) -> None:
        self.model, self.version, self.variant, self.replica = model, version, variant, replica
        self._store = store
        self._flush_rows, self._flush_seconds = flush_rows, flush_seconds
        self.sample_rate = sample_rate
        self._clock, self._rng = clock, rng
        self._rows: list[dict[str, Any]] = []
        self._last_flush = clock()

    def add(self, stats: pd.DataFrame, output: pd.DataFrame, request_id: str) -> None:
        """Buffer each row with probability `sample_rate`; flush when a limit is reached."""
        ts = int(self._clock() * 1000)
        for i, (prediction, confidence) in enumerate(
            zip(output["prediction"], output["confidence"], strict=True)
        ):
            if self._rng() >= self.sample_rate:
                continue
            row = {
                "ts": ts,
                "request_id": request_id,
                "model": self.model,
                "version": self.version,
                "variant": self.variant,
                "prediction": str(prediction),
                "confidence": float(confidence),
            }
            row.update({f"stat_{c}": float(stats[c].iloc[i]) for c in stats.columns})
            self._rows.append(row)
        if len(self._rows) >= self._flush_rows or self._clock() - self._last_flush >= self._flush_seconds:
            self.flush()

    def flush(self) -> str | None:
        """Write the buffered rows; the written URI, or None when there was nothing (or the write failed)."""
        rows, self._rows = self._rows, []
        now = self._clock()
        self._last_flush = now
        if not rows:
            return None
        day = datetime.fromtimestamp(now, UTC).date()
        name = f"{self.variant}-v{self.version}-{self.replica}-{int(now * 1000)}.parquet"
        uri = f"{inference_log_prefix(self.model, day)}{name}"
        table = pa.Table.from_pylist(rows)
        table = table.set_column(
            0, "ts", pa.array(table.column("ts").to_pylist(), pa.timestamp("ms", tz="UTC"))
        )
        buffer = io.BytesIO()
        pq.write_table(table, buffer)
        try:
            self._store.write_bytes(uri, buffer.getvalue())
        except Exception as err:  # never fail a request because of the log
            _log.warning(
                "inference_log.flush_failed", uri=uri, rows=len(rows), error=f"{type(err).__name__}: {err}"
            )
            return None
        return uri
