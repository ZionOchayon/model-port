"""FR-14: every Prometheus target is up, every required scrape pool has a target, every source has data.

Usage: uv run python scripts/metrics_check.py --profile smoke|full [--negative]
Port-forwards Prometheus itself (localhost:19090, stopped at exit). Run it after the e2e flow or `make demo`,
when every source has produced data; `--negative` also requires `airflow_dagrun_failed_count`, which exists
once a DAG run has failed (CI starts one on purpose). Exit 0 when everything passes, else 1 with each problem.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Final

import requests

PROMETHEUS: Final = "http://localhost:19090"
PORT_FORWARD: Final = [
    *("kubectl", "--context", "kind-modelport", "port-forward"),
    *("-n", "monitoring", "svc/kps-prometheus", "19090:9090"),
]
# prometheus-operator pools are <kind>/<namespace>/<name>/<endpoint index>
REQUIRED_POOLS: Final = (
    "podMonitor/monitoring/ray-head/0",
    "podMonitor/monitoring/ray-workers/0",
    "serviceMonitor/ray/kuberay-operator/0",
    "serviceMonitor/monitoring/airflow-statsd/0",
    "serviceMonitor/monitoring/pushgateway/0",
    "serviceMonitor/monitoring/mlflow/0",
    "serviceMonitor/monitoring/minio/0",
    "serviceMonitor/monitoring/kps-kube-state-metrics/0",
    "serviceMonitor/monitoring/kps-kubelet/1",  # cAdvisor
    "serviceMonitor/monitoring/kps-prometheus-node-exporter/0",
)
# one query per source: each must return at least one sample now
QUERIES: Final = (
    "ray_node_cpu_count",
    "kuberay_cluster_info",
    "ray_serve_deployment_request_counter_total",
    "ray_modelport_prediction_confidence_count",
    "ray_modelport_router_requests_total",
    "airflow_scheduler_heartbeat",
    "airflow_dagrun_duration_count",
    "modelport_drift_samples",
    "modelport_alias_version",
    "kube_pod_status_ready",
    "container_cpu_usage_seconds_total",
)
NEGATIVE_QUERIES: Final = ("airflow_dagrun_failed_count",)


def target_problems(targets: Iterable[Mapping[str, Any]], required: Sequence[str]) -> list[str]:
    """Each active target that is not up, and each required pool with no target."""
    targets = list(targets)
    problems = [
        f"target {t['scrapePool']} ({t.get('scrapeUrl', '?')}) is {t['health']}: {t.get('lastError') or '-'}"
        for t in targets
        if t["health"] != "up"
    ]
    pools = {t["scrapePool"] for t in targets}
    return problems + [f"no active target in scrape pool {p}" for p in required if p not in pools]


def data_problems(samples: Mapping[str, int]) -> list[str]:
    """Each query with no samples."""
    return [f"no samples for {q}" for q, n in samples.items() if n == 0]


def _get(path: str, **params: str) -> Any:  # noqa: ANN401 - Prometheus JSON
    response = requests.get(f"{PROMETHEUS}{path}", params=params, timeout=15)
    response.raise_for_status()
    return response.json()["data"]


def _wait_ready(deadline_s: float = 30.0) -> None:
    end = time.monotonic() + deadline_s
    while True:
        try:
            if requests.get(f"{PROMETHEUS}/-/ready", timeout=2).ok:
                return
        except requests.RequestException:
            pass
        if time.monotonic() > end:
            raise SystemExit("metrics_check: Prometheus port-forward did not become ready in 30 s")
        time.sleep(1)


def run(negative: bool) -> list[str]:
    """Both checks against the forwarded Prometheus."""
    problems = target_problems(_get("/api/v1/targets", state="active")["activeTargets"], REQUIRED_POOLS)
    queries = (*QUERIES, *NEGATIVE_QUERIES) if negative else QUERIES
    samples = {q: len(_get("/api/v1/query", query=q)["result"]) for q in queries}
    return problems + data_problems(samples)


def main(argv: Sequence[str] | None = None) -> int:
    """Port-forward, check, print each problem; exit 1 when there is any."""
    parser = argparse.ArgumentParser(description="FR-14 metrics check")
    parser.add_argument("--profile", choices=("smoke", "full"), required=True)
    parser.add_argument("--negative", action="store_true", help="also require airflow_dagrun_failed_count")
    args = parser.parse_args(argv)
    forward = subprocess.Popen(PORT_FORWARD, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)  # noqa: S603
    try:
        _wait_ready()
        problems = run(args.negative)
    finally:
        forward.terminate()
    for problem in problems:
        print(f"metrics_check: {problem}")
    print(f"metrics_check: profile {args.profile}: {len(problems)} problems")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
