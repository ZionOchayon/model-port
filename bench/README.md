# bench

`make bench` (FR-13) measures one batch job run in two ways, then runs the three failure drills once each.
The README table and `docs/results/` come only from this script.

## What is measured

The job is the `covertype-classifier` champion scoring a fixed 50,000-row Covertype input (bench day
`2000-01-01`), submitted exactly as the `batch_inference` DAG submits it. Each run is one of:

| Pattern       | How it runs                                                                                                            |
| ------------- | ---------------------------------------------------------------------------------------------------------------------- |
| `rayjob`      | A RayJob with its own cluster: the shared cluster's head and `batch` worker group, fixed at one worker, deleted after. |
| `shared`      | A Ray Jobs API submission to `modelport-shared` after its workers have scaled to zero (what a daily job sees).         |
| `shared-warm` | The same submission right after a `shared` run, while its worker is still up.                                          |

Each rep runs `shared`, then `shared-warm`, then waits for the shared workers to reach zero, then runs `rayjob`.
Only one job runs at a time, so the patterns never compete for the node.

- **Startup:** from submission (RayJob created, or `submit_job` returned) until the job's driver reports
  `RUNNING`, polled every second. For `rayjob` this includes creating the cluster; for `shared` the driver
  starts on the running head, so waiting for a worker counts as duration.
- **Duration:** from `RUNNING` until the job succeeds.
- **CPU-seconds:** `increase(container_cpu_usage_seconds_total)` in Prometheus over every container of the pods
  that ran the job, from submission to 30 s after the end. For `rayjob` that is its cluster and its submitter
  pod; for `shared` it is the head (with its autoscaler) and the worker, because the idle head is part of
  what the pattern costs.

The table shows the median and the range of n runs per pattern, the host, and a link to the raw CSV.

## Running it

```bash
make up PROFILE=smoke && make ui
set -a; . .modelport/credentials.env; set +a
make bench REPS=5
```

The first run onboards both example models as the CI end-to-end flow does; models that already have a
champion are skipped. Results:

- `docs/results/raw/bench-<UTC>.csv`: one row per run (`rep, pattern, startup_s, duration_s, cpu_s, job_id,
started_at_utc, host`).
- `docs/results/table.md` and the README block, written by `bench/summarize.py` from the newest CSV.
  `python bench/summarize.py --check README.md` fails when either one does not match it.
- `docs/results/drills.md`: one line per drill (`kill_worker`, `zero_downtime`, `canary_failure`) with the
  `scripts/verify.py` check that judged it.

The `bench` GitHub workflow runs the same steps on one runner (manually, or on a push to `bench-run/**`) and
uploads the results as an artifact. `python bench/bench.py --dry-run-check` prints every submission, RayJob and
query without a cluster.
