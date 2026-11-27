"""Constant-rate load against `/<model>/predict`: canary traffic and the zero-downtime drill.

Requests are paced on a fixed schedule; a slow response delays the next request but never causes a
burst to make up missed slots. Every outcome is counted by HTTP status (`conn` for connection errors).
"""

import statistics
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import pandas as pd

from modelport.clients.serve import ServeClient
from modelport.errors import ModelportError, PlatformUnavailableError
from modelport.serving import encode_rows
from modelport.spec import ColumnSpec


@dataclass(frozen=True)
class LoadResult:
    """Totals of one load run."""

    sent: int
    ok: int
    failed: int
    status_counts: Mapping[str, int]
    latency_ms_p50: float | None
    latency_ms_p95: float | None
    variants: Mapping[str, int]  # successful responses per variant


def _percentile(values: Sequence[float], q: int) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    return statistics.quantiles(values, n=100, method="inclusive")[q - 1]


def run_load(
    client: ServeClient,
    model: str,
    rows: pd.DataFrame,
    *,
    spec_inputs: Sequence[ColumnSpec],
    rps: float,
    duration_s: float,
    batch_rows: int = 8,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> LoadResult:
    """Send `batch_rows`-row requests at `rps` for `duration_s` seconds, cycling over `rows`."""
    frame = rows.reset_index(drop=True)
    batches = [
        encode_rows(frame.iloc[i : i + batch_rows], spec_inputs) for i in range(0, len(frame), batch_rows)
    ]
    interval = 1.0 / rps
    statuses: Counter[str] = Counter()
    variants: Counter[str] = Counter()
    latencies: list[float] = []
    start = clock()
    next_at = start
    sent = 0
    while next_at - start < duration_s - 1e-9:  # tolerance: summed intervals drift below the duration
        wait = next_at - clock()
        if wait > 0:
            sleep(wait)
        began = clock()
        try:
            response = client.predict(model, batches[sent % len(batches)])
        except PlatformUnavailableError:
            statuses["conn"] += 1
        except ModelportError as err:
            statuses[str(err.details.get("status", "error"))] += 1
        else:
            statuses["200"] += 1
            variants[response.variant] += 1
            latencies.append((clock() - began) * 1000)
        sent += 1
        next_at = max(next_at + interval, clock())  # no burst catch-up after a slow response
    ok = statuses["200"]
    return LoadResult(
        sent=sent,
        ok=ok,
        failed=sent - ok,
        status_counts=dict(statuses),
        latency_ms_p50=_percentile(latencies, 50),
        latency_ms_p95=_percentile(latencies, 95),
        variants=dict(variants),
    )
