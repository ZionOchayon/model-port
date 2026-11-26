# ADR-0015: Two Python distributions in a uv workspace with a `src` layout

- Status: Accepted
- Date: 2026-11-14

## Context

The data scientist's SDK and the platform are separate layers. A top-level package named `platform` would shadow Python's standard library module.

## Decision

`sdk/` builds `modelport` (SDK + CLI); `platform/` builds `modelport-platform` (import `modelport_platform`, which depends on `modelport`). One `uv.lock` at the root. DAG files stay in `platform/dags/`.

## Consequences

- The data scientist installs only `modelport`.
- Ray, Serve and gate code live under `platform/src/modelport_platform/`.
