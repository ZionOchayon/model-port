# ADR-0010: Airflow is the only writer of aliases and of the RayService

- Status: Accepted
- Date: 2026-11-14

## Context

The CLI's `promote` and `rollback`, the gate and the canary all change aliases and serving. Concurrent writers could leave aliases and serving out of sync.

## Decision

Only Airflow tasks write aliases and patch the RayService. The CLI triggers DAGs (`serve_sync`) instead of writing itself. RayService writes run in a one-slot pool `serve_config`, and the whole Serve config is rendered from registry state (ADR-0018).

## Consequences

- The CLI needs only Airflow and MLflow credentials, never Kubernetes access.
- Manual actions take a DAG run (about a minute of overhead) and appear in Airflow's history.
