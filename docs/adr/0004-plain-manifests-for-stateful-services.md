# ADR-0004: Plain manifests for PostgreSQL, MinIO and MLflow; Helm only where charts are maintained

- Status: Accepted
- Date: 2026-11-14

## Context

The first plan was to install MLflow and MinIO with Helm charts. The MinIO charts are frozen and reference images that cannot be pulled; the community MLflow chart lags MLflow (appVersion 3.16.0 vs 3.18.0); Bitnami's PostgreSQL images moved to a legacy repository in 2025.

## Decision

Deploy PostgreSQL (official image, StatefulSet), MinIO (local image, Deployment) and MLflow (official `-full` image, Deployment) as plain manifests in `deploy/manifests/` with a kustomize overlay per profile. Use Helm for KubeRay, Airflow, kube-prometheus-stack and the Pushgateway, whose charts are maintained by their projects.

## Consequences

- About 250 lines of YAML that the project owns and understands.
- `deploy/helm/` has values for the four charted components only; every component is still configured as code.
- Upgrading these three services means editing an image pin, not a chart.
