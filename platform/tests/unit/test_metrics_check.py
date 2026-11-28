import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
import metrics_check


def target(pool, health="up", job="j", error=""):
    return {
        "scrapePool": pool,
        "health": health,
        "labels": {"job": job},
        "lastError": error,
        "scrapeUrl": "u",
    }


def test_target_problems():
    pools = ("podMonitor/monitoring/ray-head/0", "serviceMonitor/monitoring/mlflow/0")
    good = [target(pools[0]), target(pools[1])]
    assert metrics_check.target_problems(good, pools) == []
    bad = [target(pools[0], health="down", error="connection refused"), target("other/0")]
    assert metrics_check.target_problems(bad, pools) == [
        "target podMonitor/monitoring/ray-head/0 (u) is down: connection refused",
        "no active target in scrape pool serviceMonitor/monitoring/mlflow/0",
    ]


def test_data_problems():
    results = {"ray_node_cpu_count": 3, "kuberay_cluster_info": 0}
    assert metrics_check.data_problems(results) == ["no samples for kuberay_cluster_info"]


def test_every_pool_and_query_is_pinned():
    assert len(metrics_check.REQUIRED_POOLS) == 10
    assert "modelport_drift_samples" in metrics_check.QUERIES
    assert "airflow_dagrun_failed_count" in metrics_check.NEGATIVE_QUERIES
