# Serving: one RayService for every model

Every model with a champion is served by one RayService, `modelport-serve`, in namespace `ray`. Airflow is its only writer: each change renders the whole Serve config from the registry and applies it, so the same registry state always gives the same config (FR-9).

## Topology

For each model the config has a router app and one app per served version:

| App                  | Route              | What it runs                                                                                                                  |
| -------------------- | ------------------ | ----------------------------------------------------------------------------------------------------------------------------- |
| `<model>`            | `/<model>`         | `ModelRouter`: answers `POST /<model>/predict` and `GET /<model>/healthz`, and splits traffic between champion and candidate. |
| `<model>--champion`  | none (router only) | `ModelDeployment` for the champion version.                                                                                   |
| `<model>--candidate` | none (router only) | `ModelDeployment` for the candidate, only while a canary runs.                                                                |

The router sends each request to the candidate with probability `candidate_weight` (`serving.canary_weight` during a canary, 0 otherwise, 1 for a model's first version, which has no champion). There is no fallback from a failing candidate to the champion: the canary has to see the candidate's errors to reject it. The one exception is a candidate app Serve has not built yet, for the few seconds after a config change: its requests go to the champion.

Serve listens on port 8000. `make ui` forwards it to `localhost:8000`.

## Request and response

```bash
modelport predict covertype-classifier rows.parquet --rows 5
```

```text
POST /<model>/predict
{"instances": [{"elevation": 2596, "aspect": 51, ...}, ...]}          1 to 1,024 rows

200 {"model": "covertype-classifier", "version": "3", "variant": "champion",
     "predictions": [{"prediction": "Spruce/Fir", "confidence": 0.91}, ...]}
```

| Status | When                                                                                                                                                                        |
| ------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `200`  | Every row was scored. `version` and `variant` say which version answered.                                                                                                   |
| `404`  | Any path other than `POST /<model>/predict` and `GET /<model>/healthz`.                                                                                                     |
| `422`  | The body is not JSON, is not a valid `PredictRequest`, or does not fit the signature (missing or extra column, wrong type). The message names the column and how to fix it. |
| `500`  | The version failed while scoring (`"error": "candidate failed: ..."`). Counted as an error by the canary.                                                                   |

Binary columns (images) travel as base64 strings; `modelport predict` encodes a `.png` or `.jpg` file for a model with exactly one binary input. JSON, CSV and Parquet files are read as rows. The request and response types are `modelport.serving.PredictRequest` and `PredictResponse`, shared by Serve and the CLI.

## One runtime environment per version

Each version app has its own Ray `runtime_env`: the `pip` packages the version was trained with (its `modelport.runtime_pip` tag, from `runtime.pip` in `model.yaml`) and the platform's environment variables. Two models with different dependencies are served side by side (check V-09), and a candidate can use different packages from its champion.

Replicas, CPUs and memory per replica come from the model's `resources.serve`. The router takes 0.1 CPU.

## Startup self-test

A version replica loads the MLflow model, then runs it on the `example/input.parquet` saved with its training run. If the signature check fails, the replica raises and Serve never marks the app running: a broken version cannot take traffic, and the DAG that deployed it fails fast with the app's message instead of waiting.

## What each replica records

- **Metrics** (scraped by Prometheus): requests and latency per variant from the router; predictions per class, confidence and the drift baseline from each version.
- **Inference log:** a sample of request rows (`drift.inference_log_sample_rate`, default all) buffered in the replica and written to `s3://inference-logs/<model>/<YYYY-MM-DD>/<variant>-v<version>-<replica>-<epoch ms>.parquet` with `ts, request_id, model, version, variant, prediction, confidence` and the model's `input_stats` columns. A failed write is logged and the rows dropped; it never fails a request. The daily drift check (v0.3) reads these files.

## Who changes what is served

| Change                               | Started by                                                                               |
| ------------------------------------ | ---------------------------------------------------------------------------------------- |
| A canary, then promotion or rollback | `deploy_candidate`, after a version passes the gate ([canary.md](canary.md))             |
| A manual promotion or rollback       | `modelport promote` / `modelport rollback` through `serve_sync` ([canary.md](canary.md)) |

Both write through the Airflow pool `serve_config` (one slot), so two changes never overwrite each other's config.
