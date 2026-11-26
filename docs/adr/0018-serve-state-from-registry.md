# ADR-0018: Render the whole RayService from registry state

- Status: Accepted
- Date: 2026-11-14

## Context

Several DAGs change serving (canary, promote, rollback) for different models. Incremental edits of `serveConfigV2` from each DAG would race and drift.

## Decision

Serving intent lives in MLflow: the `champion` alias plus registered-model tags `modelport.serve.candidate_version`, `modelport.serve.candidate_weight`, `modelport.serve.fault_error_rate`. Every write renders the complete `serveConfigV2` from that state (pure function, sorted) and applies it in pool `serve_config`.

## Consequences

- Applying the same state is a no-op; reruns are safe.
- One DAG's canary survives another model's promotion, because both render from the same state.
