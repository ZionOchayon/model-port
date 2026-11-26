# ADR-0013: Evaluation and batch jobs share one RayCluster through the Jobs API

- Status: Accepted
- Date: 2026-11-14

## Context

Training runs in RayJobs and batch inference on a shared RayCluster. Evaluation needs the model's dependencies (torch, xgboost), which the Airflow image does not have.

## Decision

Evaluation jobs also run on `modelport-shared` via `JobSubmissionClient`, with deterministic `submission_id`s and the version's `runtime_env`. The cluster autoscales workers from 0.

## Consequences

- No framework dependencies in the Airflow image.
- Candidate and champion are measured on the same cluster and worker sizing.
- Evaluation queues behind batch jobs (pool `shared_cluster`).
