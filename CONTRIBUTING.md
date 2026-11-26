# Contributing to modelport

## Setup

You need Docker. Everything else runs in the dev toolbox (ADR-0019):

```bash
docker compose build toolbox
docker compose run --rm toolbox make help
```

On a Linux host with uv, kind, kubectl and Helm installed, you can run the same `make` targets directly. `scripts/install-tools.sh` installs the pinned kind, kubectl and Helm with sha256 checks.

```bash
uv sync --all-packages --group dev
```

## Commands

| Command          | What it does                                            |
| ---------------- | ------------------------------------------------------- |
| `make lint`      | `ruff check` and `ruff format --check`                  |
| `make typecheck` | `mypy --strict` on both packages                        |
| `make test`      | unit, contract and local integration tests (no cluster) |
| `make help`      | every target                                            |

Inside the toolbox, prefix each command with `docker compose run --rm toolbox`.

## Rules

- **Versions** come only from `versions.env`. No `latest` tags, no unpinned charts. Change a pin in one PR that runs the smoke CI.
- **Commits** follow [Conventional Commits](https://www.conventionalcommits.org/): `feat(sdk): add ModelSpec`, `ci: add smoke job`, `docs: …`. Scopes are top-level folders. Update `CHANGELOG.md` under `## [Unreleased]` in the same commit.
- **Tests** are never skipped, loosened or given longer timeouts to make a change pass. Fix the cause.
- **Names are contracts.** Module paths, CLI flags, metric names, aliases, tag keys, S3 paths and DAG IDs change everywhere at once, or not at all.
- **Decisions** that shape the architecture get an ADR in `docs/adr/` (Context, Decision, Consequences) and a line in `docs/adr/README.md`.
- **Data**: public datasets only, with credit in `NOTICE`.
