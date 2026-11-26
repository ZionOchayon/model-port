# ADR-0001: Record architecture decisions

- Status: Accepted
- Date: 2026-11-14

## Context

modelport is built by an implementation agent that sees only this repository. Decisions made during design must be readable later, with their reasons, so they are not reopened by accident.

## Decision

Every significant decision is an ADR in `docs/adr/NNNN-title.md` with Context, Decision and Consequences. Status is `Accepted` unless superseded; a superseding ADR links back. The index is `docs/adr/README.md`.

## Consequences

- Reviewers can see why a version, tool or pattern was chosen.
- Changing a decision means writing a new ADR, which costs a few minutes and forces a reason.
