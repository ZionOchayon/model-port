# ADR-0017: Pushgateway groups keyed by model and source, replaced on every push

- Status: Accepted
- Date: 2026-11-14

## Context

The Pushgateway never forgets series, so per-run labels create stale series (Prometheus docs, "When to use the Pushgateway").

## Decision

Jobs `modelport_drift`, `modelport_lifecycle`, `modelport_jobs`; grouping keys `model` (+ `source`, `pattern`, `kind`); pushes use PUT (`push_to_gateway`) so each push replaces its group; no instance, run ID or timestamp labels. Timestamps are metric values.

## Consequences

- Exactly one current series per model and feature.
- A skipped drift run removes stale PSI values (the PUT has no PSI series).
