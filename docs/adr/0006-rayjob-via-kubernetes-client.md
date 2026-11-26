# ADR-0006: Create RayJobs with the Kubernetes client and wait with a reschedule-mode sensor

- Status: Accepted
- Date: 2026-11-14

## Context

Airflow must create a RayJob per training run and wait for it. Options: the Astronomer Ray provider (archived 2026-07-01, manages RayClusters not RayJobs), `KubernetesPodOperator` with kubectl, the cncf provider's `KubernetesCreateResourceOperator` (no 409 handling, so retries fail), or the Kubernetes client with a custom deferrable trigger.

## Decision

`ensure()` creates the RayJob or adopts it on 409; a `@task.sensor(mode="reschedule", poke_interval=15)` reads `status.jobDeploymentStatus`/`jobStatus`.

## Consequences

- No extra dependency and no trigger class to maintain; retries are safe.
- Reschedule mode frees the executor slot between pokes; a deferrable trigger over `kubernetes_asyncio` is the documented upgrade path if slots become scarce.
- Status resolution is 15 s, so DAG-side startup timings are coarse; `make bench` measures at 1 s.
