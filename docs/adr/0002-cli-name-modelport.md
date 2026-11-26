# ADR-0002: Name the CLI and SDK `modelport` (not `onboard`)

- Status: Accepted
- Date: 2026-11-14

## Context

An early draft proposed the command `onboard` and the import `from onboard import BaseModel`, and leaves the name open. The repository, the platform and the metric prefix are all `modelport`. `onboard` describes only `submit`; `status`, `promote`, `rollback` and `predict` are not onboarding.

## Decision

The command is `modelport` and the SDK package is `modelport` (`from modelport import BaseModel`). Reason: one name for repo, package, command and metrics, so a data scientist who reads `modelport` anywhere knows which tool it is.

## Consequences

- The SDK lives in `sdk/src/modelport/`.
- Metric names (`modelport_*`), tags (`modelport.*`) and the CLI share one prefix.
- The README explains the rename in one line.
