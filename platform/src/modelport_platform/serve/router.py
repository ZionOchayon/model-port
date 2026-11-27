"""One HTTP entry point per model (`/<model>/predict`) that splits traffic between champion and candidate.

The split is a weighted coin per request. There is no fallback from a failing candidate to the champion:
the canary must see the candidate's errors to reject it. Only a candidate app that Serve has not built yet
(no handle) sends the request to the champion.
"""

import json
import random
import time
from typing import Any, Literal

from ray import serve
from ray.serve.exceptions import RayServeException
from starlette.requests import Request
from starlette.responses import JSONResponse

from modelport.logs import get_logger
from modelport_platform.metrics import names

_log = get_logger(__name__)
Variant = Literal["champion", "candidate"]


def choose_variant(weight: float, draw: float) -> Variant:
    """`candidate` when `draw` (uniform in [0, 1)) falls below `weight`."""
    return "candidate" if draw < weight else "champion"


def _get_app_handle(app_name: str) -> Any:  # noqa: ANN401 - serve.handle.DeploymentHandle
    return serve.get_app_handle(app_name)


class RouterMetrics:
    """Router metrics of one model (exported as `ray_<name>`)."""

    def __init__(self, model: str) -> None:
        from ray.util import metrics

        self.model = model
        self._requests = metrics.Counter(
            names.ROUTER_REQUESTS,
            description="Requests by variant and outcome.",
            tag_keys=("model", "variant", "outcome"),
        )
        self._latency = metrics.Histogram(
            names.ROUTER_LATENCY_MS,
            description="Request latency through the router.",
            boundaries=list(names.ROUTER_LATENCY_BUCKETS_MS),
            tag_keys=("model", "variant"),
        )
        self._threshold = metrics.Gauge(
            names.LATENCY_THRESHOLD_MS,
            description="serving.latency_p95_ms of the model.",
            tag_keys=("model",),
        )

    def request(self, variant: str, outcome: str) -> None:
        """Count one request: outcome `ok`, `client_error` or `error`."""
        self._requests.inc(1, tags={"model": self.model, "variant": variant, "outcome": outcome})

    def latency(self, variant: str, ms: float) -> None:
        """Observe one request's latency."""
        self._latency.observe(ms, tags={"model": self.model, "variant": variant})

    def set_latency_threshold(self, ms: float) -> None:
        """The p95 latency the `ServeLatencyP95` alert compares against."""
        self._threshold.set(ms, tags={"model": self.model})


@serve.deployment(name="ModelRouter", num_replicas=1, max_ongoing_requests=64)
class ModelRouter:
    """Routes `POST /<model>/predict` to the champion or candidate app; `GET /<model>/healthz`."""

    def __init__(self, model: str) -> None:
        self.model = model
        self.metrics = RouterMetrics(model)
        self.champion_app: str | None = f"{model}--champion"
        self.candidate_app: str | None = None
        self.candidate_weight = 0.0
        self._handles: dict[str, Any] = {}

    def reconfigure(self, config: dict[str, Any]) -> None:
        """Apply `user_config`; cached handles of apps no longer named are dropped.

        No candidate app means weight 0; no champion app (a model's first version) means weight 1.
        """
        self.champion_app = config.get("champion_app") or None
        self.candidate_app = config.get("candidate_app") or None
        weight = float(config.get("candidate_weight", 0.0))
        self.candidate_weight = 0.0 if not self.candidate_app else 1.0 if not self.champion_app else weight
        current = {self.champion_app, self.candidate_app}
        self._handles = {name: h for name, h in self._handles.items() if name in current}
        self.metrics.set_latency_threshold(float(config.get("latency_threshold_ms", 500.0)))
        _log.info(
            "router.reconfigured",
            model=self.model,
            candidate_app=self.candidate_app,
            weight=self.candidate_weight,
        )

    def _handle(self, app_name: str) -> Any:  # noqa: ANN401 - DeploymentHandle
        if app_name not in self._handles:
            self._handles[app_name] = _get_app_handle(app_name)
        return self._handles[app_name]

    def _route(self, variant: Variant) -> tuple[Variant, Any]:
        """The variant that serves the request and its handle.

        The candidate app arrives in the same config change as its weight, and the router is reconfigured
        seconds before Serve has built that app: until it exists, the champion serves.
        """
        app_name = self.candidate_app if variant == "candidate" else self.champion_app
        assert app_name is not None  # weight 0 without a candidate app, 1 without a champion app
        try:
            return variant, self._handle(app_name)
        except RayServeException:
            if variant == "champion" or self.champion_app is None:
                raise
            _log.info("router.candidate_not_ready", model=self.model, candidate_app=app_name)
            return "champion", self._handle(self.champion_app)

    async def __call__(self, request: Request) -> JSONResponse:
        """Dispatch by path: `/predict`, `/healthz`, anything else 404."""
        path = request.url.path.removeprefix(f"/{self.model}").rstrip("/") or "/"
        if path == "/healthz" and request.method == "GET":
            self._handle(self.champion_app or self.candidate_app or "")
            return JSONResponse({"model": self.model, "status": "ok"})
        if path != "/predict" or request.method != "POST":
            return JSONResponse(
                {"error": f"no route {request.method} {request.url.path}; use POST /{self.model}/predict"},
                status_code=404,
            )
        variant = choose_variant(self.candidate_weight, random.random())  # noqa: S311 - traffic split
        start = time.perf_counter()
        try:
            body = json.loads(await request.body())
        except json.JSONDecodeError as err:
            self.metrics.request(variant, "client_error")
            return JSONResponse({"error": f"request body is not JSON: {err.msg}"}, status_code=422)
        try:
            variant, handle = self._route(variant)
            result: dict[str, Any] = await handle.predict.remote(body)
        except Exception as err:  # no fallback: the canary must see the candidate's errors
            self.metrics.request(variant, "error")
            _log.warning("router.variant_failed", model=self.model, variant=variant, error=type(err).__name__)
            return JSONResponse({"error": f"{variant} failed: {type(err).__name__}"}, status_code=500)
        finally:
            self.metrics.latency(variant, (time.perf_counter() - start) * 1000)
        if result.get("status") == 422:
            self.metrics.request(variant, "client_error")
            return JSONResponse({"error": result["error"]}, status_code=422)
        self.metrics.request(variant, "ok")
        return JSONResponse(result)


def build_router_app(args: dict[str, str]) -> serve.Application:
    """Serve `import_path` target: `{"model"}` from the rendered config."""
    application: serve.Application = ModelRouter.bind(args["model"])  # type: ignore[attr-defined]
    return application
