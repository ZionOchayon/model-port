"""Serve application for one pinned model version (alias resolved once, by the DAG that renders the config).

The replica refuses to start when the version fails its signature self-test on its own example input,
so a broken version makes its application DEPLOY_FAILED instead of serving errors.
"""

import asyncio
import random
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from pydantic import ValidationError
from ray import serve

from modelport.errors import ModelportError, SignatureMismatchError
from modelport.logs import get_logger
from modelport.naming import Tag
from modelport.pyfunc import ModelportPyfunc, load_version
from modelport.registry import MlflowRegistry, registry_from_env
from modelport.serving import PredictionItem, PredictRequest, PredictResponse, decode_request
from modelport.signature import check_input_frame
from modelport_platform.drift.baseline import Baseline
from modelport_platform.ports import DatasetStore
from modelport_platform.serve.inference_log import InferenceLogBuffer
from modelport_platform.serve.metrics import ServeMetrics

_log = get_logger(__name__)
Variant = Literal["champion", "candidate"]


def _registry() -> MlflowRegistry:
    return registry_from_env()


def _load_pyfunc(model_uri: str) -> ModelportPyfunc:
    return load_version(model_uri)


def _store() -> DatasetStore:
    from modelport_platform.s3 import S3DatasetStore, boto3_client
    from modelport_platform.settings import PlatformSettings

    return S3DatasetStore(boto3_client(PlatformSettings.from_env()))


def _replica_id() -> str:
    try:
        return str(serve.get_replica_context().replica_id.unique_id)
    except Exception:  # outside a Serve replica (tests, local runs)
        return "local"


def _unprocessable(message: str) -> dict[str, Any]:
    return {"error": message, "status": 422}


@serve.deployment(
    name="ModelDeployment", max_ongoing_requests=16, graceful_shutdown_timeout_s=20, health_check_period_s=10
)
class ModelDeployment:
    """One model version behind the router: predict, metrics, inference log."""

    def __init__(self, model: str, version: str, variant: Variant) -> None:
        start = time.monotonic()
        self.model, self.version, self.variant = model, version, variant
        registry = _registry()
        info = registry.get_version(model, version)
        self.pyfunc = _load_pyfunc(registry.model_uri(model, version))
        train_run = info.tags[Tag.TRAIN_RUN_ID]
        with tempfile.TemporaryDirectory() as tmp:
            example = pd.read_parquet(registry.download(train_run, "example/input.parquet", Path(tmp)))
        try:
            self.pyfunc.model.run_inference(example)
        except ModelportError as err:
            _log.error("serve.self_test_failed", model=model, version=version, error=err.what)
            raise RuntimeError(f"signature self-test failed for {model} v{version}: {err.what}") from err
        baseline = Baseline.from_json(registry.read_json(train_run, "drift/baseline.json"))
        self.metrics = ServeMetrics(model, version)
        self.metrics.set_baseline(baseline.confidence_mean, baseline.class_shares)
        self.metrics.set_model_info(variant)
        self.fault_error_rate = 0.0
        self.log = InferenceLogBuffer(
            model=model, version=version, variant=variant, store=_store(), replica=_replica_id()
        )
        _log.info(
            "serve.replica_started",
            model=model,
            version=version,
            variant=variant,
            load_seconds=round(time.monotonic() - start, 3),
        )

    def reconfigure(self, config: dict[str, Any]) -> None:
        """Apply `user_config`: the drill's fault rate, log sampling and live drift thresholds."""
        self.fault_error_rate = float(config.get("fault_error_rate", 0.0))
        self.log.sample_rate = float(config.get("inference_log_sample_rate", 1.0))
        if "live_confidence_drop" in config or "live_class_share_increase" in config:
            self.metrics.set_live_thresholds(
                confidence_drop=float(config.get("live_confidence_drop", 0.15)),
                class_share_increase=float(config.get("live_class_share_increase", 0.30)),
            )

    async def predict(self, request: dict[str, Any]) -> dict[str, Any]:
        """`PredictRequest` -> `PredictResponse`; a bad request returns `{"error", "status": 422}`."""
        model = self.pyfunc.model
        try:
            parsed = PredictRequest.model_validate(request)
        except ValidationError as err:
            return _unprocessable(
                f"not a valid request ({err.error_count()} errors): {err.errors()[0]['msg']}"
            )
        try:
            frame = decode_request(parsed, model.spec.signature.inputs)
            check_input_frame(frame, model.spec)
        except SignatureMismatchError as err:
            return _unprocessable(f"{err.what}. {err.fix}")
        if self.fault_error_rate > 0 and random.random() < self.fault_error_rate:  # noqa: S311 - drill only
            raise RuntimeError("fault injection")
        output = await asyncio.to_thread(model.run_inference, frame)
        stats = model.input_stats(frame)
        self.metrics.observe(output, stats)
        self.log.add(stats, output, uuid.uuid4().hex)
        items = [
            PredictionItem(prediction=str(p), confidence=float(c))
            for p, c in zip(output["prediction"], output["confidence"], strict=True)
        ]
        response = PredictResponse(
            model=self.model, version=self.version, variant=self.variant, predictions=items
        )
        return response.model_dump()

    async def check_health(self) -> None:
        """Healthy once constructed: the self-test already ran."""

    def __del__(self) -> None:
        """Write the rows still buffered."""
        log = getattr(self, "log", None)
        if log is not None:
            log.flush()


def build_model_app(args: dict[str, str]) -> serve.Application:
    """Serve `import_path` target: `{"model", "version", "variant"}` from the rendered config."""
    variant: Variant = "candidate" if args["variant"] == "candidate" else "champion"
    application: serve.Application = ModelDeployment.bind(args["model"], args["version"], variant)  # type: ignore[attr-defined]
    return application
