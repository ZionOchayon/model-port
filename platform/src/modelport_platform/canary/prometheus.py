"""Prometheus adapter (`MetricsQuery`) and the canary's PromQL over Ray Serve's built-in metrics."""

import math
from collections.abc import Mapping
from datetime import datetime
from typing import Any

import requests

from modelport.errors import PlatformUnavailableError
from modelport_platform.canary.core import CanaryStats
from modelport_platform.ports import MetricsQuery


class PrometheusClient:
    """`GET /api/v1/query` (instant queries)."""

    def __init__(
        self, base_url: str, *, timeout_s: float = 15.0, session: requests.Session | None = None
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self._timeout_s = timeout_s
        self._session = session or requests.Session()

    def _query(self, promql: str, at: datetime | None) -> list[dict[str, Any]]:
        params: dict[str, Any] = {"query": promql}
        if at is not None:
            params["time"] = at.timestamp()
        try:
            response = self._session.get(
                f"{self.base_url}/api/v1/query", params=params, timeout=self._timeout_s
            )
        except requests.RequestException as err:
            raise PlatformUnavailableError(
                what=f"Cannot reach Prometheus at {self.base_url} ({type(err).__name__})",
                fix="Check the kps-prometheus pod in namespace monitoring.",
            ) from err
        payload = response.json() if response.status_code < 500 else {}
        if response.status_code >= 400 or payload.get("status") != "success":
            raise PlatformUnavailableError(
                what=f"Prometheus query failed (HTTP {response.status_code}): {payload.get('error', '')}",
                fix=f"Check the query and Prometheus at {self.base_url}: {promql}",
            )
        result: list[dict[str, Any]] = payload["data"]["result"]
        return result

    def scalar(self, promql: str, *, at: datetime | None = None) -> float | None:
        """One aggregated value; None for an empty result or NaN.

        Raises:
            PlatformUnavailableError: Prometheus is unreachable or the query failed.
            ValueError: the query returned more than one series (it must aggregate).
        """
        result = self._query(promql, at)
        if not result:
            return None
        if len(result) > 1:
            raise ValueError(f"expected one aggregated value, got {len(result)} series: {promql}")
        value = float(result[0]["value"][1])
        return None if math.isnan(value) else value

    def vector(self, promql: str, *, at: datetime | None = None) -> list[tuple[Mapping[str, str], float]]:
        """(labels, value) per series."""
        return [(r["metric"], float(r["value"][1])) for r in self._query(promql, at)]


def canary_queries(app: str, window_s: int) -> dict[str, str]:
    """Requests, errors and p95 latency of one Serve application over the last `window_s` seconds."""
    selector = f'{{application="{app}"}}[{window_s}s]'
    return {
        "requests": f"sum(increase(ray_serve_deployment_request_counter_total{selector}))",
        "errors": f"sum(increase(ray_serve_deployment_error_counter_total{selector}))",
        "p95_ms": "histogram_quantile(0.95, sum by (le) (increase("
        f"ray_serve_deployment_processing_latency_ms_bucket{selector})))",
    }


def canary_stats(query: MetricsQuery, app: str, window_s: int, *, at: datetime) -> CanaryStats:
    """The app's stats at `at`; an empty result is 0 for counts and None for p95."""
    q = canary_queries(app, window_s)
    return CanaryStats(
        requests=query.scalar(q["requests"], at=at) or 0.0,
        errors=query.scalar(q["errors"], at=at) or 0.0,
        p95_ms=query.scalar(q["p95_ms"], at=at),
    )
