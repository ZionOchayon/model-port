# ADR-0003: Keep MinIO, built from source at its last community release

- Status: Accepted
- Date: 2026-11-14

## Context

MinIO is the chosen S3 store. On 2026-11-14 the `minio/minio` repository is archived, the community edition is source-only, and no official image can be pulled from Docker Hub or Quay. The upstream Dockerfile starts `FROM minio/minio:latest`, which no longer exists.

## Decision

Build MinIO from source at `RELEASE.2025-10-15T17-29-55Z` (the last community release; it also fixes a privilege-escalation advisory) with a two-stage Dockerfile (pinned Go builder, pinned Alpine runtime). The image is local only (`modelport/minio:<release>`), cached in CI. Buckets are created by a Python Job, not `mc` (whose image is also gone). Every component uses only the S3 API, so the fallback is SeaweedFS (Apache-2.0, maintained).

## Consequences

- About 3 minutes of Go build on a cold cache; zero on a warm one.
- No security fixes will come for MinIO; acceptable for a local development stack, stated in the README.
- Switching to SeaweedFS is revisited if the source build ever fails.
