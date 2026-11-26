# ADR-0008: Package code once: a zip for training, an MLflow pyfunc for everything after

- Status: Accepted
- Date: 2026-11-14

## Context

Training needs the data scientist's code on every Ray worker. Serving, batch and evaluation need code plus weights with a signature.

## Decision

`modelport submit` uploads a deterministic `code.zip` (single top-level dir) to a submission run in MLflow; the RayJob uses its S3 location as `runtime_env.working_dir`. Registration logs an MLflow pyfunc (`ModelportPyfunc`) with `code_paths=[model dir]`, the weights as artifacts and the signature from `model.yaml`; Serve, batch and evaluation load that pyfunc by version.

## Consequences

- The data scientist never handles S3 credentials (MLflow proxies uploads).
- One loading path after training, so evaluation measures what Serve runs.
- Depends on Ray's support for `s3://` `working_dir` with `AWS_ENDPOINT_URL`; if that fails, the train entrypoint downloads the zip itself.
