# Evaluation gate

A newly trained version must pass the gate before it can be served. The Airflow DAG `evaluate_and_promote` evaluates the new version, and the current champion, on the same validation set, and decides. A passed version becomes `candidate`; a failed one becomes `rejected` with every reason it failed.

## The rules

For a model's **first version** (no champion yet) there is one condition:

- the primary metric (`evaluation.metric`) ≥ `evaluation.min_absolute`.

**With a champion**, all three must hold:

1. **Better overall:** the primary metric ≥ the champion's + `evaluation.min_improvement`.
2. **No class left behind:** no class's score drops more than `evaluation.max_drop_per_class` below the champion's.
3. **Not slower:** inference latency ≤ the champion's × (1 + `evaluation.max_latency_increase`).

The validation set is the champion's (so both are judged on the same rows), or the new version's own `val.parquet` for a first version. Latency is the median of `evaluation.latency_repeats` timed calls of `run_inference` on `evaluation.latency_batch_size` rows, after one warm-up call.

Every failed condition is reported, not only the first:

```text
f1_macro 0.790 < champion 0.800 + min_improvement 0.000
class 'Shirt' dropped 0.081 (0.800 -> 0.719) > 0.050
latency 12.1 ms > champion 10.0 ms + 20%
```

## DAG `evaluate_and_promote`

Started by `train_model` with run ID `evaluate__<model>__v<N>`; params `model` and `version`.

| Task                 | What it does                                                                                                                                        |
| -------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| `prepare`            | Picks the validation set and the evaluation runs. Skips (a rerun is safe) when the version is already decided.                                      |
| `evaluate_candidate` | Runs the evaluation job for the new version on the shared Ray cluster `modelport-shared` (pool `shared_cluster`), with the version's `runtime.pip`. |
| `evaluate_champion`  | The same for the champion, or skipped when there is none.                                                                                           |
| `decide`             | Applies the rules, sets the state and tags, and pushes the result to the Pushgateway.                                                               |
| `should_deploy`      | Continues to the canary only for a passed version with `serving.auto_deploy` (from v0.2).                                                           |

The evaluation job (`python -m modelport_platform.gate.evaluate`) loads the version through MLflow, predicts the validation set in batches, and logs `gate/report.json` to an evaluation run: the primary metric, every class's score (F1 for `f1_macro`, recall for `accuracy`), the median latency with its batch size, the row count and the validation URI.

## What it records

| Where                         | What                                                                                                  |
| ----------------------------- | ----------------------------------------------------------------------------------------------------- |
| Version tag `modelport.state` | `candidate` or `rejected`                                                                             |
| `modelport.gate.result`       | `passed` or `failed`                                                                                  |
| `modelport.gate.reasons`      | JSON list of every condition with its numbers                                                         |
| `modelport.gate.eval_run_id`  | The evaluation run with `gate/report.json` (and `gate/champion_report.json` when there is a champion) |
| Alias `candidate`             | Set on a passed version                                                                               |

`modelport status <model>` shows all of this, with the new version's report next to the champion's; `modelport submit --wait` exits 8 with the reasons when the version is rejected.

## Version states

`registered` → `candidate` or `rejected` at the gate. From v0.2 a candidate goes through the canary to `champion`; the replaced champion becomes `previous`, and older versions `retired`. Aliases `champion`, `candidate` and `previous` always point at versions in those states.
