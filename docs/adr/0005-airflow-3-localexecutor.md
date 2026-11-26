# ADR-0005: Airflow 3.4 on chart 1.22.0 with LocalExecutor, SimpleAuthManager and DAGs baked into the image

- Status: Accepted
- Date: 2026-11-14

## Context

Airflow 3.4.0 is current; the newest released chart (1.22.0) defaults to 3.2.2. The cluster has 16 GB. The CLI must trigger DAGs over REST.

## Decision

Run Airflow 3.4.0 (`airflowVersion: "3.4.0"`) on chart 1.22.0 with `LocalExecutor` (tasks run in the scheduler pod), `SimpleAuthManager` with one user whose password comes from a generated Secret, the triggerer for deferrable sensors, and DAGs copied into the image (no git-sync, no PVC). StatsD metrics through the chart's exporter.

## Consequences

- Lowest memory and fewest moving parts that still run deferrable sensors.
- Risk: chart 1.22.0 was not released against 3.4; the fallback is Airflow 3.2.2, the chart's tested default.
- New DAG code means a new image; Airflow 3's DAG versioning keeps running runs on their version.
