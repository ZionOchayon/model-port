"""The gate decision (pure core): absolute bar for a first model, else metric, per-class and latency checks.

All checks against a champion run, so every failing reason is reported, and an average improvement never
hides a class that got worse.
"""

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any, Self

from modelport.spec import EvaluationSpec, MetricName

_TOLERANCE = 1e-9  # float noise must not flip a decision at the exact limit


@dataclass(frozen=True)
class EvalReport:
    """One version evaluated on one validation set (`gate/report.json`)."""

    model: str
    version: str
    val_uri: str
    n_samples: int
    metric_name: MetricName
    metric: float
    per_class: Mapping[str, float]
    latency_ms_median: float
    latency_batch_size: int
    created_at: str  # ISO 8601 UTC

    def to_json(self) -> dict[str, Any]:
        """JSON form."""
        payload = asdict(self)
        payload["metric_name"] = str(self.metric_name)
        payload["per_class"] = dict(self.per_class)
        return payload

    @classmethod
    def from_json(cls, payload: Mapping[str, Any]) -> Self:
        """Read `to_json`."""
        return cls(
            model=payload["model"],
            version=str(payload["version"]),
            val_uri=payload["val_uri"],
            n_samples=int(payload["n_samples"]),
            metric_name=MetricName(payload["metric_name"]),
            metric=float(payload["metric"]),
            per_class={k: float(v) for k, v in payload["per_class"].items()},
            latency_ms_median=float(payload["latency_ms_median"]),
            latency_batch_size=int(payload["latency_batch_size"]),
            created_at=payload["created_at"],
        )


@dataclass(frozen=True)
class GatePolicy:
    """The `evaluation` block of model.yaml."""

    metric: MetricName
    min_improvement: float
    max_drop_per_class: float
    max_latency_increase: float
    min_absolute: float

    @classmethod
    def from_spec(cls, spec: EvaluationSpec) -> Self:
        """Policy from the spec."""
        return cls(
            metric=spec.metric,
            min_improvement=spec.min_improvement,
            max_drop_per_class=spec.max_drop_per_class,
            max_latency_increase=spec.max_latency_increase,
            min_absolute=spec.min_absolute,
        )


class CheckName(StrEnum):
    """The gate's checks."""

    ABSOLUTE = "absolute"
    METRIC = "metric"
    PER_CLASS = "per_class"
    LATENCY = "latency"


@dataclass(frozen=True)
class CheckResult:
    """One check's outcome with a human-readable detail."""

    name: CheckName
    passed: bool
    detail: str


@dataclass(frozen=True)
class GateDecision:
    """All checks and the overall result."""

    passed: bool
    checks: tuple[CheckResult, ...]

    @property
    def reasons(self) -> tuple[str, ...]:
        """Details of the failed checks."""
        return tuple(c.detail for c in self.checks if not c.passed)


def _absolute(candidate: EvalReport, policy: GatePolicy) -> CheckResult:
    passed = candidate.metric >= policy.min_absolute - _TOLERANCE
    op = ">=" if passed else "<"
    detail = f"{candidate.metric_name} {candidate.metric:.3f} {op} min_absolute {policy.min_absolute:.3f}"
    return CheckResult(CheckName.ABSOLUTE, passed, detail)


def _metric(candidate: EvalReport, champion: EvalReport, policy: GatePolicy) -> CheckResult:
    passed = candidate.metric >= champion.metric + policy.min_improvement - _TOLERANCE
    op = ">=" if passed else "<"
    detail = (
        f"{candidate.metric_name} {candidate.metric:.3f} {op} champion {champion.metric:.3f}"
        f" + min_improvement {policy.min_improvement:.3f}"
    )
    return CheckResult(CheckName.METRIC, passed, detail)


def _per_class(candidate: EvalReport, champion: EvalReport, policy: GatePolicy) -> CheckResult:
    over = []
    for label, before in champion.per_class.items():
        after = candidate.per_class.get(label, 0.0)
        if before - after > policy.max_drop_per_class + _TOLERANCE:
            over.append(
                f"class '{label}' dropped {before - after:.3f} ({before:.3f} -> {after:.3f})"
                f" > {policy.max_drop_per_class:.3f}"
            )
    detail = "; ".join(over) or f"no class dropped more than {policy.max_drop_per_class:.3f}"
    return CheckResult(CheckName.PER_CLASS, not over, detail)


def _latency(candidate: EvalReport, champion: EvalReport, policy: GatePolicy) -> CheckResult:
    limit = champion.latency_ms_median * (1 + policy.max_latency_increase)
    passed = candidate.latency_ms_median <= limit + _TOLERANCE
    op = "<=" if passed else ">"
    detail = (
        f"latency {candidate.latency_ms_median:.1f} ms {op} champion {champion.latency_ms_median:.1f} ms"
        f" + {policy.max_latency_increase:.0%}"
    )
    return CheckResult(CheckName.LATENCY, passed, detail)


def decide(candidate: EvalReport, champion: EvalReport | None, policy: GatePolicy) -> GateDecision:
    """Gate decision for `candidate` against the `champion` (None for a model's first version).

    Raises:
        ValueError: the report's metric is not the policy's, or the two reports used different
            validation sets (programming errors: the DAG evaluates both on the same set).
    """
    if candidate.metric_name != policy.metric:
        raise ValueError(f"report metric {candidate.metric_name} is not the policy metric {policy.metric}")
    if champion is None:
        checks: tuple[CheckResult, ...] = (_absolute(candidate, policy),)
    else:
        if candidate.val_uri != champion.val_uri:
            raise ValueError(f"reports use different val sets: {candidate.val_uri} vs {champion.val_uri}")
        checks = (
            _metric(candidate, champion, policy),
            _per_class(candidate, champion, policy),
            _latency(candidate, champion, policy),
        )
    return GateDecision(passed=all(c.passed for c in checks), checks=checks)
