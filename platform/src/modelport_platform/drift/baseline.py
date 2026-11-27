"""The drift baseline stored with each version as `drift/baseline.json`: bins and shares, never raw rows."""

import math
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Final

import numpy as np
import pandas as pd

from modelport import BaseModel
from modelport_platform.drift.psi import categorical_shares, quantile_edges, shares

BASELINE_MAX_ROWS: Final = 5000
_JSON_INF: Final = 1e308  # JSON has no infinity


class FeatureKind(StrEnum):
    """How a feature is binned."""

    CONTINUOUS = "continuous"
    CATEGORICAL = "categorical"


@dataclass(frozen=True)
class FeatureBaseline:
    """One feature's bins and baseline shares."""

    name: str
    kind: FeatureKind
    edges: tuple[float, ...] | None  # continuous: len = bins + 1, outer edges -inf/+inf
    categories: tuple[str, ...] | None  # categorical (last bin "__other__" implicit)
    shares: tuple[float, ...]
    mean: float | None  # continuous only

    def to_json(self) -> dict[str, Any]:
        """JSON form (infinite edges as +-1e308)."""
        payload: dict[str, Any] = {"kind": str(self.kind), "shares": list(self.shares)}
        if self.edges is not None:
            payload["edges"] = [max(-_JSON_INF, min(_JSON_INF, e)) for e in self.edges]
            payload["mean"] = self.mean
        if self.categories is not None:
            payload["categories"] = list(self.categories)
        return payload

    @classmethod
    def from_json(cls, name: str, payload: Mapping[str, Any]) -> "FeatureBaseline":
        """Read back `to_json` (+-1e308 edges become +-inf)."""
        edges = payload.get("edges")
        categories = payload.get("categories")
        return cls(
            name=name,
            kind=FeatureKind(payload["kind"]),
            edges=tuple(math.copysign(math.inf, e) if abs(e) >= _JSON_INF else float(e) for e in edges)
            if edges is not None
            else None,
            categories=tuple(categories) if categories is not None else None,
            shares=tuple(float(s) for s in payload["shares"]),
            mean=payload.get("mean"),
        )


@dataclass(frozen=True)
class Baseline:
    """Every drift feature of one model version."""

    model: str
    version: str
    n_rows: int
    features: Mapping[str, FeatureBaseline]
    confidence_mean: float
    class_shares: Mapping[str, float]

    def to_json(self) -> dict[str, Any]:
        """The `drift/baseline.json` payload."""
        return {
            "model": self.model,
            "version": self.version,
            "n_rows": self.n_rows,
            "confidence_mean": self.confidence_mean,
            "class_shares": dict(self.class_shares),
            "features": {name: f.to_json() for name, f in self.features.items()},
        }

    @classmethod
    def from_json(cls, payload: Mapping[str, Any]) -> "Baseline":
        """Read `drift/baseline.json`."""
        return cls(
            model=payload["model"],
            version=str(payload["version"]),
            n_rows=int(payload["n_rows"]),
            features={n: FeatureBaseline.from_json(n, f) for n, f in payload["features"].items()},
            confidence_mean=float(payload["confidence_mean"]),
            class_shares={k: float(v) for k, v in payload["class_shares"].items()},
        )


def _continuous(name: str, values: np.ndarray, bins: int) -> FeatureBaseline:
    edges = quantile_edges(values, bins)
    return FeatureBaseline(
        name=name,
        kind=FeatureKind.CONTINUOUS,
        edges=tuple(float(e) for e in edges),
        categories=None,
        shares=tuple(float(s) for s in shares(values, edges)),
        mean=float(np.mean(values)),
    )


def build_baseline(
    model: BaseModel, val: pd.DataFrame, *, model_name: str, version: str, bins: int
) -> Baseline:
    """Bin every `input_stats` column, `confidence` and `prediction` over the first 5,000 `val` rows.

    `model` is loaded; the split is already a random permutation, so the first rows are a fair sample.
    """
    sample = val.head(BASELINE_MAX_ROWS)
    output = model.run_inference(sample)
    stats = model.input_stats(sample)
    features = {f"stat_{c}": _continuous(f"stat_{c}", stats[c].to_numpy(np.float64), bins) for c in stats}
    confidence = output["confidence"].to_numpy(np.float64)
    features["confidence"] = _continuous("confidence", confidence, bins)
    labels = tuple(model.spec.labels)
    predicted = categorical_shares(output["prediction"].astype(str).tolist(), labels)
    features["prediction"] = FeatureBaseline(
        name="prediction",
        kind=FeatureKind.CATEGORICAL,
        edges=None,
        categories=labels,
        shares=tuple(float(s) for s in predicted),
        mean=None,
    )
    return Baseline(
        model=model_name,
        version=version,
        n_rows=len(sample),
        features=features,
        confidence_mean=float(np.mean(confidence)),
        class_shares=dict(zip(labels, (float(s) for s in predicted[:-1]), strict=True)),
    )
