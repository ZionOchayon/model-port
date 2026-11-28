"""Daily drift: PSI of the champion's logged traffic against its baseline, per feature (FR-16).

`load_inference_logs` reads the inference log files of a time window; `compute_drift` is pure and takes the
rows, so the same check runs on any frame.
"""

import io
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Final

import numpy as np
import pandas as pd

from modelport.logs import get_logger
from modelport.naming import inference_log_prefix
from modelport_platform.drift.baseline import Baseline
from modelport_platform.drift.psi import feature_psi
from modelport_platform.ports import DatasetStore

MAX_DRIFT_ROWS: Final = 50_000
_log = get_logger(__name__)


@dataclass(frozen=True)
class DriftResult:
    """One model's drift over a window."""

    model: str
    version: str
    samples: int
    psi: dict[str, float]  # empty when skipped
    skipped: bool
    reason: str | None


def _days(start: datetime, end: datetime) -> list[date]:
    return [start.date() + timedelta(days=i) for i in range((end.date() - start.date()).days + 1)]


def load_inference_logs(store: DatasetStore, model: str, start: datetime, end: datetime) -> pd.DataFrame:
    """Every logged row of `model` with `start < ts <= end` (empty frame when there are none)."""
    frames = [
        pd.read_parquet(io.BytesIO(store.read_bytes(uri)))
        for day in _days(start, end)
        for uri in store.list(inference_log_prefix(model, day))
        if uri.endswith(".parquet")
    ]
    if not frames:
        return pd.DataFrame()
    rows = pd.concat(frames, ignore_index=True)
    return rows[(rows["ts"] > start) & (rows["ts"] <= end)].reset_index(drop=True)


def compute_drift(logs: pd.DataFrame, baseline: Baseline, *, version: str, min_samples: int) -> DriftResult:
    """PSI of every baseline feature over the rows of `version`; skipped below `min_samples` rows."""
    rows = logs[logs["version"].astype(str) == version] if not logs.empty else logs
    n = len(rows)
    if n < min_samples:
        _log.info("drift.skipped", model=baseline.model, version=version, samples=n, min_samples=min_samples)
        return DriftResult(baseline.model, version, n, {}, True, f"{n} rows < min_samples {min_samples}")
    if n > MAX_DRIFT_ROWS:
        rows = rows.sample(MAX_DRIFT_ROWS, random_state=0)
    psi: dict[str, float] = {}
    for name, feature in baseline.features.items():
        if name not in rows:
            _log.warning("drift.feature_missing", model=baseline.model, feature=name)
            continue
        column = rows[name]
        values = (
            column.astype(str).tolist() if feature.categories is not None else column.to_numpy(np.float64)
        )
        psi[name] = feature_psi(feature, values)
    worst = max(psi, key=lambda f: psi[f], default=None)
    _log.info(
        "drift.computed",
        model=baseline.model,
        samples=n,
        max_psi=psi[worst] if worst else None,
        max_feature=worst,
    )
    return DriftResult(baseline.model, version, n, psi, False, None)
