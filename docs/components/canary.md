# Canary, promotion and rollback

A version that passed the gate does not replace the champion at once. It is served beside the champion with a share of live traffic, judged on its live errors and latency, then promoted with zero downtime or rolled back and rejected with its reasons (FR-10, FR-11). Two DAGs change what is served, and both use the same swap.

## DAG `deploy_candidate`

`evaluate_and_promote` starts it when a version passes the gate and its `serving.auto_deploy` is `true` (the default), with run ID `deploy__<model>__v<version>`. It can also be triggered by hand with `{"model", "version"}`.

| Param                 | Default                                       | Effect                                                                                                                   |
| --------------------- | --------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| `model`, `version`    | required                                      | The version to canary. It must hold the alias `candidate` in state `candidate`, otherwise the run is skipped.            |
| `observation_minutes` | `serving.observation_window_minutes`          | Length of the observation window (1 to 120).                                                                             |
| `synthetic_traffic`   | `MODELPORT_CANARY_SYNTHETIC_TRAFFIC` (`true`) | Send 5 requests a second from the version's validation rows for the whole window, so a local stack has traffic to judge. |
| `fault_error_rate`    | `0`                                           | Drill only: the share of the candidate's requests that fail on purpose.                                                  |

```text
check_candidate → start_canary → wait_canary_ready → canary_traffic ∥ observe → judge → branch
  passed: route_all_to_candidate → wait_route → promote_aliases → update_champion_app
          → wait_champion_ready → retire_candidate_app → wait_final
  failed: remove_candidate_app → mark_rejected → wait_after_rollback
```

1. **Start.** The candidate is written to the registered model's serve tags with its weight (`serving.canary_weight`, or 1 when the model has no champion yet) and the RayService is rendered and applied. `wait_canary_ready` waits until every app runs. If the candidate app cannot deploy (for example its self-test fails), the canary fails at once with the app's message instead of waiting out the window.
2. **Observe.** `observe` waits for the window to pass. Traffic is the router's real split: no request falls back to the champion.
3. **Judge.** Prometheus is read over the window, and every rule below is checked.
4. **Promote or roll back.** A failed canary is an outcome, not a failed run: the run succeeds and the reasons are stored.

## The rules

All of them are checked, and every failed rule is reported with its numbers.

| Rule                                                                                                                | Limit from                         |
| ------------------------------------------------------------------------------------------------------------------- | ---------------------------------- |
| At least `min_canary_requests` requests reached the candidate.                                                      | `serving.min_canary_requests` (50) |
| Candidate error rate at most `max_error_rate`.                                                                      | `serving.max_error_rate` (0.02)    |
| Candidate error rate at most the champion's + 0.01, when the champion served at least `min_canary_requests` itself. | fixed margin                       |
| The candidate has latency data, and its p95 is at most `latency_p95_ms`.                                            | `serving.latency_p95_ms` (500 ms)  |
| Candidate p95 at most the champion's p95 × (1 + `max_latency_increase`).                                            | `evaluation.max_latency_increase`  |

A model's first version has no champion, so only the absolute limits apply. Errors are the requests the candidate's app failed on (Ray Serve's error counter, answered `500` by the router); a `422` for bad input is a normal answer and does not count. The verdict is stored on the version as `modelport.canary.result` (`passed` or `failed`) and `modelport.canary.reasons` (a JSON list), shown on the version in MLflow; a rejected version's state is in `modelport status NAME`.

## The zero-downtime swap

Every request must land on an app that is running. Updating the champion app in place would restart its replicas under traffic, so a promotion moves traffic first and changes apps only while they are idle:

| Step                     | Router                 | `<model>--champion`                          | `<model>--candidate`     |
| ------------------------ | ---------------------- | -------------------------------------------- | ------------------------ |
| Canary                   | weight `canary_weight` | old version                                  | new version              |
| `route_all_to_candidate` | weight 1               | old version, idle                            | new version, all traffic |
| `promote_aliases`        |                        | aliases move: new `champion`, old `previous` |                          |
| `update_champion_app`    | weight 1               | new version, starting                        | new version, all traffic |
| `retire_candidate_app`   | weight 0               | new version, all traffic                     | removed                  |

Each step waits until the RayService reports every app running before the next one. A rollback only removes the candidate app; the champion never stopped serving. Every change goes through the Airflow pool `serve_config` (one slot), and each step is safe to retry.

## Manual `promote` and `rollback`: DAG `serve_sync`

```bash
modelport promote covertype-classifier 4 --wait
modelport rollback covertype-classifier --yes
```

`modelport promote NAME VERSION` and `modelport rollback NAME` start `serve_sync` with `{"model", "action": "promote"|"rollback", "version"}`. It refuses (and the CLI exits 8) when a canary is running for the model, when the version did not pass the gate or is `rejected`, when it is already the champion, or, for a rollback, when there is no `previous` version.

```text
plan → branch
  served:     stage_target → wait_staged → route_all_to_target → wait_routed → apply_aliases
  not served: apply_aliases
apply_aliases → update_champion_app → wait_champion_ready → retire_target_app → wait_final
```

The target is first staged as `<model>--candidate` with no traffic, then the same swap runs as for a canary. A model that is not served yet only moves its aliases. A rollback makes `previous` the champion again; the version it replaces is retired and the model has no `previous` until the next promotion.

## Drills

Both run in the `bench` workflow and their outcomes are in [docs/results/drills.md](../results/drills.md).

| Drill                                    | What it does                                                                                                                                                                                       | Check |
| ---------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----- |
| `bench/drills/canary_failure.sh [MODEL]` | Trains a new version that passes the gate, then canaries it with `fault_error_rate` 0.5 for 3 minutes. It must be rolled back and `rejected` for its error rate, and the champion must not change. | V-10  |
| `bench/drills/zero_downtime.sh [MODEL]`  | Trains a new version, then runs its canary and promotion while a load generator sends 5 requests a second for 7 minutes. No request may fail, and the promotion must finish while the load runs.   | V-11  |

The load generator also runs on its own: `python -m modelport_platform.tools.loadgen --model NAME --rows FILE` exits 1 on any failed request.
