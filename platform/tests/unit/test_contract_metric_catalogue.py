"""The metric catalogue is the contract between the code, the rules and the dashboards.

Every custom metric is a constant in `metrics.names`, and every metric name the rules (and dashboards) use is
a custom metric as exported, a documented built-in, or a recording rule.
"""

import json
import re
from pathlib import Path

import yaml

from modelport_platform.metrics import names

ROOT = Path(__file__).resolve().parents[3]

# name in code -> exported names (Ray adds `ray_`, counters get `_total`, histograms their three series)
SERVE_HISTOGRAMS = {names.PREDICTION_CONFIDENCE, names.INPUT_STAT, names.ROUTER_LATENCY_MS}
SERVE_COUNTERS = {names.PREDICTIONS, names.ROUTER_REQUESTS}
SERVE_GAUGES = {
    names.BASELINE_CONFIDENCE_MEAN,
    names.BASELINE_CLASS_SHARE,
    names.LIVE_DRIFT_THRESHOLD,
    names.MODEL_INFO,
    names.LATENCY_THRESHOLD_MS,
}
PUSHED = {
    names.PSI,
    names.PSI_THRESHOLD,
    names.DRIFT_SAMPLES,
    names.DRIFT_LAST_SUCCESS,
    names.GATE_LAST_RESULT,
    names.GATE_LAST_RESULT_TIMESTAMP,
    names.CANARY_LAST_RESULT,
    names.CANARY_LAST_RESULT_TIMESTAMP,
    names.ALIAS_VERSION,
    names.JOB_STARTUP,
    names.JOB_DURATION,
    names.JOB_CPU,
}
# built-ins the rules and dashboards may use: Ray, KubeRay, Airflow statsd, kube-state-metrics, cAdvisor
BUILTIN_PREFIXES = ("ray_serve_", "ray_node_", "ray_object_store_", "ray_tasks", "kuberay_", "airflow_")
BUILTIN = {
    "kube_pod_status_ready",
    "kube_pod_container_status_restarts_total",
    "kube_pod_container_status_last_terminated_reason",
    "container_cpu_usage_seconds_total",
    "container_memory_working_set_bytes",
    "ALERTS",
    "up",
}
METRIC = re.compile(r"\b(?:ray|modelport|airflow|kuberay|kube|container)[a-zA-Z0-9_:]*|\bALERTS\b")
# label matchers and grouping clauses hold label names, not metric names
LABELS = re.compile(r"\{[^}]*\}|\b(?:by|without|on|ignoring|group_left|group_right)\s*\([^)]*\)")


def exported() -> set[str]:
    out = {f"ray_{n}_total" for n in SERVE_COUNTERS} | {f"ray_{n}" for n in SERVE_GAUGES}
    out |= {f"ray_{n}_{s}" for n in SERVE_HISTOGRAMS for s in ("bucket", "sum", "count")}
    return out | PUSHED


def test_every_catalogue_name_is_a_constant():
    constants = {v for k, v in vars(names).items() if k.isupper() and isinstance(v, str)}
    jobs = {names.JOB_DRIFT, names.JOB_LIFECYCLE, names.JOB_JOBS}
    assert constants - jobs == SERVE_HISTOGRAMS | SERVE_COUNTERS | SERVE_GAUGES | PUSHED


def used_names() -> tuple[set[str], set[str]]:
    """(metric names used in expressions, names recorded by rules) across the rules and dashboards."""
    rules = yaml.safe_load((ROOT / "deploy/monitoring/prometheus-rules.yaml").read_text(encoding="utf-8"))
    exprs = [r["expr"] for g in rules["spec"]["groups"] for r in g["rules"]]
    recorded = {r["record"] for g in rules["spec"]["groups"] for r in g["rules"] if "record" in r}
    for path in sorted((ROOT / "deploy/monitoring/dashboards").glob("*.json")):
        exprs += re.findall(r'"expr":\s*"((?:[^"\\]|\\.)*)"', path.read_text(encoding="utf-8"))
        json.loads(path.read_text(encoding="utf-8"))  # every dashboard is valid JSON
    used = {m for e in exprs for m in METRIC.findall(LABELS.sub(" ", e))}
    return used, recorded


def test_every_used_name_is_in_the_catalogue():
    used, recorded = used_names()
    known = exported() | recorded | BUILTIN
    unknown = sorted(n for n in used if n not in known and not n.startswith(BUILTIN_PREFIXES))
    assert unknown == []
    assert used & PUSHED, "the rules read pushed metrics"
