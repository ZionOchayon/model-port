# ADR-0009: Aliases are the only selector, resolved once per deployment or job

- Status: Accepted
- Date: 2026-11-14

## Context

Requirements: loading code always selects models by alias, never by version number, and a batch job keeps the version it started with. Serve applications only redeploy when their config changes.

## Decision

Every consumer resolves an alias to a version exactly once (the RayService render, the batch `plan` task, the evaluation `prepare` task), records it, and loads `models:/<name>/<version>`. The version is part of the Serve app args, so an alias change produces a config change and a controlled redeploy.

## Consequences

- No consumer silently switches versions mid-run.
- The rule "select by alias" holds; the version number appears only as the recorded result of that selection.
