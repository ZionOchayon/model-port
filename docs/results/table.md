# Bench results (FR-13)

Method: `bench/README.md`.

| Pattern | n | Startup s | Duration s | CPU-seconds |
| --- | --- | --- | --- | --- |
| RayJob (cluster per job) | 5 | 27.6 (27.6-28.6) | 11.8 (11.8-15.0) | 51.2 (48.7-53.4) |
| Shared cluster, workers from zero | 5 | 1.0 (1.0-3.0) | 27.3 (22.2-29.3) | 48.8 (43.8-57.7) |
| Shared cluster, worker already up | 5 | 1.0 (1.0-1.0) | 8.1 (8.1-9.1) | 11.7 (4.0-22.8) |

Median (min-max) of n runs per pattern. Host: runner: 4 vCPU, 15 GiB. Raw data: [bench-20261128T082637Z.csv](raw/bench-20261128T082637Z.csv).
