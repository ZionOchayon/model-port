# ADR-0016: Install Prometheus in v0.2, not v0.3

- Status: Accepted
- Date: 2026-11-14

## Context

The first plan put kube-prometheus-stack in v0.3, but the v0.2 canary compares errors and latency in Prometheus, and the v0.2 pattern benchmark measures CPU-seconds.

## Decision

Install kube-prometheus-stack in v0.2 (without Grafana and Alertmanager in smoke); v0.3 adds the Pushgateway, ServiceMonitors, rules, Alertmanager config and dashboards.

## Consequences

- v0.2 can be verified on its own.
- v0.3's scope shrinks to monitoring content rather than infrastructure.
