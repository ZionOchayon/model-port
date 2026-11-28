import csv
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "bench"))
import bench
import summarize

ROOT = Path(__file__).resolve().parents[3]

SHARED = yaml.safe_load((ROOT / "deploy/manifests/base/shared-raycluster.yaml").read_text(encoding="utf-8"))
JOB = {
    "submission_id": "batch__covertype-classifier__bench-rayjob-1",
    "entrypoint": "python -m modelport_platform.batch.job --model covertype-classifier --version 1",
    "runtime_env": {"env_vars": {"MLFLOW_TRACKING_URI": "http://mlflow:5000"}, "pip": {"packages": ["x==1"]}},
    "entrypoint_num_cpus": 0,
    "metadata": {"model": "covertype-classifier"},
    "force": True,
}


def test_rayjob_copies_the_shared_cluster_with_one_worker():
    job = bench.rayjob_manifest(SHARED, name="bench-rayjob-1", job=JOB, image="ray:x")
    assert job["kind"] == "RayJob"
    assert job["metadata"] == {"name": "bench-rayjob-1", "namespace": "ray", "labels": bench.LABELS}
    spec = job["spec"]
    assert spec["entrypoint"] == JOB["entrypoint"]
    assert spec["entrypointNumCpus"] == 0
    assert spec["shutdownAfterJobFinishes"] is True
    assert yaml.safe_load(spec["runtimeEnvYAML"]) == JOB["runtime_env"]
    cluster = spec["rayClusterSpec"]
    assert "enableInTreeAutoscaling" not in cluster
    assert "autoscalerOptions" not in cluster
    (group,) = cluster["workerGroupSpecs"]
    assert (group["replicas"], group["minReplicas"], group["maxReplicas"]) == (1, 1, 1)
    shared_group = SHARED["spec"]["workerGroupSpecs"][0]
    assert (
        group["template"]["spec"]["containers"][0]["resources"]
        == (shared_group["template"]["spec"]["containers"][0]["resources"])
    )
    assert group["rayStartParams"] == shared_group["rayStartParams"]
    head = cluster["headGroupSpec"]
    assert (
        head["template"]["spec"]["containers"][0]["resources"]
        == (SHARED["spec"]["headGroupSpec"]["template"]["spec"]["containers"][0]["resources"])
    )
    assert SHARED["spec"]["workerGroupSpecs"][0]["replicas"] == 0  # the source object is not changed


def test_cpu_query_and_window():
    submitted = datetime(2026, 1, 1, tzinfo=UTC)
    ended = submitted + timedelta(seconds=100.2)
    at, window = bench.cpu_window(submitted, ended)
    assert at == ended + timedelta(seconds=30)
    assert window == 131  # submission to 30 s after the end, rounded up
    query = bench.cpu_query("modelport-shared-.*", window)
    assert query == (
        'sum(increase(container_cpu_usage_seconds_total{namespace="ray", pod=~"modelport-shared-.*",'
        ' container!=""}[131s]))'
    )


def write(path, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=bench.COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def rows(pattern, values):
    return [
        {
            "rep": i + 1,
            "pattern": pattern,
            "startup_s": s,
            "duration_s": d,
            "cpu_s": c,
            "job_id": f"j{i}",
            "started_at_utc": "2026-01-01T00:00:00Z",
            "host": "runner: 4 vCPU, 15 GiB",
        }
        for i, (s, d, c) in enumerate(values)
    ]


def test_table_median_and_range(tmp_path):
    csv_path = tmp_path / "bench-20260101T000000Z.csv"
    write(
        csv_path,
        rows("rayjob", [(40, 30, 90), (50, 20, 80), (45, 25, 100)])
        + rows("shared", [(1, 60, 50), (2, 70, 70), (3, 65, 60)])
        + rows("shared-warm", [(1, 20, 20), (1, 22, 30), (1, 21, 25)]),
    )
    table = summarize.table(summarize.load(csv_path), "raw/bench-20260101T000000Z.csv")
    assert (
        "| RayJob (cluster per job) | 3 | 45.0 (40.0-50.0) | 25.0 (20.0-30.0) | 90.0 (80.0-100.0) |" in table
    )
    assert "| Shared cluster, workers from zero | 3 | 2.0 (1.0-3.0) |" in table
    assert "| Shared cluster, worker already up | 3 |" in table
    assert "runner: 4 vCPU, 15 GiB" in table
    assert "[bench-20260101T000000Z.csv](raw/bench-20260101T000000Z.csv)" in table
    assert summarize.counts(summarize.load(csv_path)) == {"rayjob": 3, "shared": 3, "shared-warm": 3}


def test_missing_cpu_shown_as_dash(tmp_path):
    csv_path = tmp_path / "bench-x.csv"
    write(csv_path, rows("rayjob", [(1, 2, "")]))
    assert "| RayJob (cluster per job) | 1 | 1.0 (1.0-1.0) | 2.0 (2.0-2.0) | — |" in summarize.table(
        summarize.load(csv_path), "raw/bench-x.csv"
    )


def test_readme_block_replaced_and_checked(tmp_path):
    readme = tmp_path / "README.md"
    readme.write_text("# x\n\n<!-- bench:start -->\nold\n<!-- bench:end -->\n\nafter\n", encoding="utf-8")
    results = tmp_path / "docs/results"
    assert summarize.main(["--root", str(tmp_path), "--check", str(readme)]) == 1  # stale
    assert summarize.main(["--root", str(tmp_path)]) == 0
    text = readme.read_text(encoding="utf-8")
    assert "old" not in text
    assert text.endswith("<!-- bench:end -->\n\nafter\n")
    assert summarize.PLACEHOLDER in text
    assert (results / "table.md").is_file()
    assert summarize.main(["--root", str(tmp_path), "--check", str(readme)]) == 0
    (results / "raw").mkdir()
    write(results / "raw/bench-20260101T000000Z.csv", rows("rayjob", [(1, 2, 3)]))
    assert summarize.main(["--root", str(tmp_path), "--check", str(readme)]) == 1  # new results
    assert summarize.main(["--root", str(tmp_path)]) == 0
    assert "docs/results/raw/bench-20260101T000000Z.csv" in readme.read_text(encoding="utf-8")


def test_readme_without_markers_fails():
    with pytest.raises(ValueError, match="bench:start"):
        summarize.replace_block("# no markers\n", "x")


def test_dry_run_builds_every_rep(capsys):
    assert bench.main(["--reps", "2", "--dry-run-check"]) == 0
    out = capsys.readouterr().out
    assert out.count("rayjob rep") == 2
    assert out.count("shared rep") == 2
    assert out.count("shared-warm rep") == 2
    assert "container_cpu_usage_seconds_total" in out


def test_repo_readme_block_is_current():
    assert summarize.main(["--root", str(ROOT), "--check", str(ROOT / "README.md")]) == 0
