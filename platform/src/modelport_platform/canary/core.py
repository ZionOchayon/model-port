"""Canary judge (pure core): the candidate's live errors and latency against its limits and the champion's.

Every rule is evaluated and every failure reported, so a rollback names all of its reasons.
"""

from dataclasses import dataclass
from typing import Self

from modelport.spec import ModelSpec


@dataclass(frozen=True)
class CanaryStats:
    """One app's traffic over the observation window."""

    requests: float
    errors: float
    p95_ms: float | None

    @property
    def error_rate(self) -> float:
        """Errors / requests; 0 when there were no requests."""
        return self.errors / self.requests if self.requests else 0.0


@dataclass(frozen=True)
class CanaryPolicy:
    """Limits from `serving` and `evaluation.max_latency_increase`."""

    max_error_rate: float
    latency_p95_ms: float
    max_latency_increase: float
    min_requests: int
    error_rate_margin: float = 0.01

    @classmethod
    def from_spec(cls, spec: ModelSpec) -> Self:
        """The policy of one model."""
        return cls(
            max_error_rate=spec.serving.max_error_rate,
            latency_p95_ms=spec.serving.latency_p95_ms,
            max_latency_increase=spec.evaluation.max_latency_increase,
            min_requests=spec.serving.min_canary_requests,
        )


@dataclass(frozen=True)
class CanaryVerdict:
    """Pass, or rollback with every reason."""

    passed: bool
    reasons: tuple[str, ...]


def judge_canary(champion: CanaryStats | None, candidate: CanaryStats, policy: CanaryPolicy) -> CanaryVerdict:
    """Judge the candidate; `champion` is None for a model's first version (absolute limits only)."""
    reasons = []
    if candidate.requests < policy.min_requests:
        reasons.append(
            f"only {candidate.requests:.0f} requests reached the candidate in the window"
            f" (need {policy.min_requests})"
        )
    if candidate.error_rate > policy.max_error_rate:
        reasons.append(f"error rate {candidate.error_rate:.3f} > max_error_rate {policy.max_error_rate:.3f}")
    # the champion is a reference only when it served enough traffic itself
    reference = champion if champion is not None and champion.requests >= policy.min_requests else None
    if reference is not None and candidate.error_rate > reference.error_rate + policy.error_rate_margin:
        reasons.append(
            f"error rate {candidate.error_rate:.3f} > champion {reference.error_rate:.3f}"
            f" + margin {policy.error_rate_margin:.3f}"
        )
    if candidate.p95_ms is None:
        reasons.append("no latency data for the candidate in the window")
    else:
        if candidate.p95_ms > policy.latency_p95_ms:
            reasons.append(
                f"p95 latency {candidate.p95_ms:.1f} ms > latency_p95_ms {policy.latency_p95_ms:.1f} ms"
            )
        champion_p95 = champion.p95_ms if champion is not None else None
        if champion_p95 is not None and candidate.p95_ms > champion_p95 * (1 + policy.max_latency_increase):
            reasons.append(
                f"p95 latency {candidate.p95_ms:.1f} ms > champion {champion_p95:.1f} ms"
                f" + {policy.max_latency_increase:.0%}"
            )
    return CanaryVerdict(passed=not reasons, reasons=tuple(reasons))
