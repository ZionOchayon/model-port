# ADR-0011: Per-model dependencies through `runtime_env.pip` on an image with the frameworks preinstalled

- Status: Accepted
- Date: 2026-11-14

## Context

Each served model must have its own dependency environment. Installing torch per job or app from scratch costs minutes and network on every start.

## Decision

Each model's exact pins (`runtime.pip`, all `==`) go into `runtime_env.pip` for its RayJob, evaluation and batch jobs and its Serve app, with `pip_check: false`. The Ray image preinstalls the example frameworks (torch CPU, torchvision, xgboost, Pillow) at the same pins, so matching requirements resolve to installed wheels; a model that pins different versions gets its own virtualenv layer.

## Consequences

- Fast startup for the examples, real isolation when versions differ.
- The example `runtime.pip` pins must match the Ray lock (`check_versions.py` enforces it).
