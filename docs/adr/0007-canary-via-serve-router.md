# ADR-0007: Canary through a Serve-level router, not KubeRay incremental upgrade

- Status: Accepted
- Date: 2026-11-14

## Context

KubeRay 1.7.2's `NewClusterWithIncrementalUpgrade` (beta) shifts traffic between whole RayClusters on a timer through the Gateway API. modelport's canary is per model, inside one cluster, with promotion decided by metrics.

## Decision

Each model gets a router application `<model>` (`ModelRouter`) that holds handles to `<model>--champion` and `<model>--candidate` and splits by a weight in `user_config`. `upgradeStrategy.type: NewCluster` is set explicitly and only `serveConfigV2` changes, which KubeRay applies in place.

## Consequences

- No Gateway controller on the small runner.
- Weight changes call `reconfigure()` without restarts.
- Promotion uses a swap protocol (route all traffic to the candidate app, update the idle champion app, route back) that gives zero downtime without relying on Serve's rolling-update surge settings.
