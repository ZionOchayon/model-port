"""Every custom metric name of the catalogue (data contracts, section on Prometheus metrics).

Names defined in Ray code are exported with a `ray_` prefix (and counters with `_total`); the constants
here are the names as written in code.
"""

from typing import Final

# Serve (ModelDeployment, ModelRouter) - exported as ray_<name>
PREDICTION_CONFIDENCE: Final = "modelport_prediction_confidence"
PREDICTIONS: Final = "modelport_predictions"
INPUT_STAT: Final = "modelport_input_stat"
BASELINE_CONFIDENCE_MEAN: Final = "modelport_baseline_confidence_mean"
BASELINE_CLASS_SHARE: Final = "modelport_baseline_class_share"
LIVE_DRIFT_THRESHOLD: Final = "modelport_live_drift_threshold"
MODEL_INFO: Final = "modelport_model_info"
ROUTER_REQUESTS: Final = "modelport_router_requests"
ROUTER_LATENCY_MS: Final = "modelport_router_latency_ms"
LATENCY_THRESHOLD_MS: Final = "modelport_latency_threshold_ms"

# Pushgateway
PSI: Final = "modelport_psi"
PSI_THRESHOLD: Final = "modelport_psi_threshold"
DRIFT_SAMPLES: Final = "modelport_drift_samples"
DRIFT_LAST_SUCCESS: Final = "modelport_drift_last_success_timestamp_seconds"
GATE_LAST_RESULT: Final = "modelport_gate_last_result"
GATE_LAST_RESULT_TIMESTAMP: Final = "modelport_gate_last_result_timestamp_seconds"
CANARY_LAST_RESULT: Final = "modelport_canary_last_result"
CANARY_LAST_RESULT_TIMESTAMP: Final = "modelport_canary_last_result_timestamp_seconds"
ALIAS_VERSION: Final = "modelport_alias_version"
JOB_STARTUP: Final = "modelport_job_startup_seconds"
JOB_DURATION: Final = "modelport_job_duration_seconds"
JOB_CPU: Final = "modelport_job_cpu_seconds"

# Pushgateway jobs
JOB_DRIFT: Final = "modelport_drift"
JOB_LIFECYCLE: Final = "modelport_lifecycle"
JOB_JOBS: Final = "modelport_jobs"

CONFIDENCE_BUCKETS: Final[tuple[float, ...]] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0)
# Ray histograms need positive bounds; values <= 0.1 land in the first bucket
INPUT_STAT_BUCKETS: Final[tuple[float, ...]] = (
    0.1,
    0.2,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    0.8,
    0.9,
    1,
    10,
    100,
    1000,
)
ROUTER_LATENCY_BUCKETS_MS: Final[tuple[float, ...]] = (5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000)
