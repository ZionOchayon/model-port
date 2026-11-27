from datetime import UTC, datetime

import pytest

from modelport.errors import PlatformUnavailableError
from modelport_platform.canary.prometheus import PrometheusClient, canary_queries, canary_stats

AT = datetime(2026, 11, 27, 12, 0, tzinfo=UTC)


def test_queries_exact_strings():
    sel = '{application="m--candidate"}[600s]'
    assert canary_queries("m--candidate", 600) == {
        "requests": f"sum(increase(ray_serve_deployment_request_counter_total{sel}))",
        "errors": f"sum(increase(ray_serve_deployment_error_counter_total{sel}))",
        "p95_ms": "histogram_quantile(0.95, sum by (le) (increase("
        f"ray_serve_deployment_processing_latency_ms_bucket{sel})))",
    }


class Response:
    def __init__(self, status_code=200, payload=None):
        self.status_code = status_code
        self.payload = payload

    def json(self):
        return self.payload


class Session:
    def __init__(self, responses):
        self.responses = responses
        self.calls = []

    def get(self, url, params=None, timeout=None):
        self.calls.append((url, params))
        return self.responses[params["query"]] if isinstance(self.responses, dict) else self.responses


def vector(*values):
    return {
        "status": "success",
        "data": {"resultType": "vector", "result": [{"metric": {}, "value": [0, v]} for v in values]},
    }


def test_empty_vector_is_zero_and_none():
    queries = canary_queries("m--candidate", 600)
    session = Session(
        {
            queries["requests"]: Response(payload=vector()),
            queries["errors"]: Response(payload=vector()),
            queries["p95_ms"]: Response(payload=vector()),
        }
    )
    stats = canary_stats(PrometheusClient("http://prom:9090", session=session), "m--candidate", 600, at=AT)
    assert (stats.requests, stats.errors, stats.p95_ms) == (0.0, 0.0, None)
    url, params = session.calls[0]
    assert url == "http://prom:9090/api/v1/query"
    assert params["time"] == AT.timestamp()


def test_values_read_and_nan_is_none():
    queries = canary_queries("a", 60)
    session = Session(
        {
            queries["requests"]: Response(payload=vector("120")),
            queries["errors"]: Response(payload=vector("3")),
            queries["p95_ms"]: Response(payload=vector("NaN")),
        }
    )
    stats = canary_stats(PrometheusClient("http://prom:9090", session=session), "a", 60, at=AT)
    assert (stats.requests, stats.errors, stats.p95_ms) == (120.0, 3.0, None)


def test_error_status_raises():
    client = PrometheusClient("http://prom:9090", session=Session(Response(503, {"status": "error"})))
    with pytest.raises(PlatformUnavailableError):
        client.scalar("up")
    client = PrometheusClient(
        "http://prom:9090", session=Session(Response(200, {"status": "error", "error": "bad query"}))
    )
    with pytest.raises(PlatformUnavailableError, match="bad query"):
        client.scalar("up(")
    client = PrometheusClient("http://prom:9090", session=Session(Response(payload=vector("1", "2"))))
    with pytest.raises(ValueError, match="2 series"):
        client.scalar("up")
