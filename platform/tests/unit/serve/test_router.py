import asyncio
import json
import random
from typing import ClassVar

from starlette.requests import Request

from modelport_platform.serve import router

CONFIG = {
    "champion_app": "m--champion",
    "candidate_app": "m--candidate",
    "candidate_weight": 0.0,
    "latency_threshold_ms": 500,
}


class Handle:
    def __init__(self, name, result=None, error=None):
        self.name, self.result, self.error, self.calls = name, result, error, 0
        self.predict = self

    def remote(self, body):
        self.calls += 1

        async def answer():
            if self.error:
                raise self.error
            return self.result or {"model": "m", "variant": self.name, "predictions": []}

        return answer()


class Recorder:
    instances: ClassVar[list["Recorder"]] = []

    def __init__(self, model):
        self.calls = []
        Recorder.instances.append(self)

    def __getattr__(self, name):
        return lambda *args, **kwargs: self.calls.append((name, args, kwargs))


def make(monkeypatch, handles, **config):
    fetched = []

    def get_handle(name):
        fetched.append(name)
        return handles[name]

    Recorder.instances.clear()
    monkeypatch.setattr(router, "_get_app_handle", get_handle)
    monkeypatch.setattr(router, "RouterMetrics", Recorder)
    deployment = router.ModelRouter.func_or_class("m")
    deployment.reconfigure({**CONFIG, **config})
    return deployment, fetched


def request(path="/m/predict", body=None, method="POST"):
    payload = json.dumps(body if body is not None else {"instances": [{"x": 1}]}).encode()

    async def receive():
        return {"type": "http.request", "body": payload, "more_body": False}

    scope = {"type": "http", "method": method, "path": path, "headers": [], "query_string": b""}
    return Request(scope, receive)


def call(deployment, req):
    response = asyncio.run(deployment(req))
    return response.status_code, json.loads(response.body)


def test_choose_variant_boundaries():
    assert all(router.choose_variant(0.0, d) == "champion" for d in (0.0, 0.5, 0.999))
    assert all(router.choose_variant(1.0, d) == "candidate" for d in (0.0, 0.5, 0.999))
    assert router.choose_variant(0.1, 0.05) == "candidate"
    assert router.choose_variant(0.1, 0.15) == "champion"


def test_split_ratio_statistical():
    rng = random.Random(0)
    share = sum(router.choose_variant(0.1, rng.random()) == "candidate" for _ in range(10_000)) / 10_000
    assert 0.08 <= share <= 0.12


def test_weight_ignored_without_candidate_app(monkeypatch):
    champion = Handle("champion")
    deployment, _ = make(monkeypatch, {"m--champion": champion}, candidate_app=None, candidate_weight=1.0)
    for _ in range(5):
        status, body = call(deployment, request())
        assert (status, body["variant"]) == (200, "champion")
    assert champion.calls == 5


def test_candidate_error_returns_500_no_fallback(monkeypatch):
    champion = Handle("champion")
    candidate = Handle("candidate", error=RuntimeError("fault injection"))
    deployment, _ = make(
        monkeypatch, {"m--champion": champion, "m--candidate": candidate}, candidate_weight=1.0
    )
    status, body = call(deployment, request())
    assert status == 500
    assert body == {"error": "candidate failed: RuntimeError"}
    assert champion.calls == 0
    outcomes = [args for name, args, _ in Recorder.instances[0].calls if name == "request"]
    assert outcomes == [("candidate", "error")]


def test_422_passthrough(monkeypatch):
    champion = Handle("champion", result={"error": "row 0 has no column 'x'", "status": 422})
    deployment, _ = make(monkeypatch, {"m--champion": champion})
    status, body = call(deployment, request())
    assert (status, body) == (422, {"error": "row 0 has no column 'x'"})
    status, _ = call(deployment, _bad_json_request())  # the router rejects a body that is not JSON
    assert status == 422


def _bad_json_request():
    async def receive():
        return {"type": "http.request", "body": b"{not json", "more_body": False}

    scope = {"type": "http", "method": "POST", "path": "/m/predict", "headers": [], "query_string": b""}
    return Request(scope, receive)


def test_reconfigure_drops_stale_handles(monkeypatch):
    handles = {
        "m--champion": Handle("champion"),
        "m--candidate": Handle("candidate"),
        "m--next": Handle("next"),
    }
    deployment, fetched = make(monkeypatch, handles, candidate_weight=1.0)
    call(deployment, request())
    call(deployment, request())
    assert fetched == ["m--candidate"]  # cached after the first call
    deployment.reconfigure({**CONFIG, "candidate_app": "m--next", "candidate_weight": 1.0})
    status, body = call(deployment, request())
    assert (status, body["variant"]) == (200, "next")
    assert fetched == ["m--candidate", "m--next"]


def test_healthz_and_unknown_paths(monkeypatch):
    deployment, _ = make(monkeypatch, {"m--champion": Handle("champion")})
    assert call(deployment, request("/m/healthz", method="GET"))[0] == 200
    status, body = call(deployment, request("/m/other"))
    assert status == 404
    assert "predict" in body["error"]


def test_metrics_and_threshold(monkeypatch):
    deployment, _ = make(monkeypatch, {"m--champion": Handle("champion")})
    call(deployment, request())
    calls = Recorder.instances[0].calls
    assert ("set_latency_threshold", (500.0,), {}) in calls
    assert ("request", ("champion", "ok"), {}) in [(n, a, {}) for n, a, _ in calls if n == "request"]
    assert any(n == "latency" and a[0] == "champion" for n, a, _ in calls)


def test_router_metrics_accept_ray_metric_rules():
    metrics = router.RouterMetrics("m")
    metrics.request("champion", "ok")
    metrics.latency("candidate", 12.5)
    metrics.set_latency_threshold(500.0)


def test_first_version_without_champion_goes_to_candidate(monkeypatch):
    candidate = Handle("candidate")
    deployment, _ = make(monkeypatch, {"m--candidate": candidate}, champion_app=None, candidate_weight=0.1)
    for _ in range(5):
        status, body = call(deployment, request())
        assert (status, body["variant"]) == (200, "candidate")
    assert call(deployment, request("/m/healthz", method="GET"))[0] == 200
