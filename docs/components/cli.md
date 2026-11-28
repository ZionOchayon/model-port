# `modelport` CLI

The CLI is the data scientist's whole interface: check a model folder, send it to the platform, see what happened. It talks to Airflow and MLflow through the port-forwards that `make ui` opens; it never needs `kubectl`.

```bash
modelport validate examples/models/covertype-classifier
modelport submit examples/models/covertype-classifier --wait
modelport status covertype-classifier
```

## Global options

Put them before the command: `modelport --json status NAME`.

| Option      | Effect                                                                                   |
| ----------- | ---------------------------------------------------------------------------------------- |
| `--json`    | Print one JSON document instead of tables and messages (errors too: `{"error": {...}}`). |
| `--debug`   | Show stack traces and HTTP details on errors.                                            |
| `--version` | Print the modelport version and exit.                                                    |
| `--help`    | Show the help of the CLI or of a command (`modelport submit --help`).                    |

## `modelport validate DIR`

Checks a model folder locally, with no cluster, in this order, and stops at the first failure: `dir.exists`, `spec.file`, `spec.schema`, `class.import`, `class.subclass`, `class.methods`, `class.train_loop`, `example.input`, `model.load`, `model.predict`, `model.input_stats`, `model.input_stats.coverage`. Each failed check prints what is wrong and how to fix it; checks after it are shown as skipped.

| Option         | Default      | Effect                                                                                          |
| -------------- | ------------ | ----------------------------------------------------------------------------------------------- |
| `--spec FILE`  | `model.yaml` | Spec file inside `DIR`.                                                                         |
| `--no-predict` | off          | Skip loading the model and running `predict` on `example/input.parquet` (the `model.*` checks). |

## `modelport submit DIR`

Runs `validate`, packages the folder into `code.zip` (the same bytes on every OS, so the same code always has the same hash), logs it to MLflow, tags the registered model with its dataset subscription, and starts the `train_model` DAG. Training is followed by registration and the evaluation gate (see [training.md](training.md) and [gate.md](gate.md)).

Submitting the same code and spec again does not train again: it shows the existing run ("This exact code was already submitted; showing the existing run."). It still makes this the latest submission, which is the one retrains use.

| Option                | Default                                   | Effect                                                                                                 |
| --------------------- | ----------------------------------------- | ------------------------------------------------------------------------------------------------------ |
| `--spec FILE`         | `model.yaml`                              | Spec file inside `DIR`.                                                                                |
| `--dataset-version N` | `data.dataset`'s version, else the latest | Train on dataset version `vN`.                                                                         |
| `--wait`              | off                                       | Block until the new version is a `candidate` (exit 0) or `rejected` (exit 8, with the gate's reasons). |
| `--timeout SECONDS`   | `3600`                                    | With `--wait`, give up after this long (exit 12); the run keeps going.                                 |

With `--json`, `submit` prints `{"model", "code_sha256", "submission_run_id", "train_run_id", "train_url", "existed"}`, plus `"outcome"` with `--wait`.

## `modelport status NAME`

Shows the model's aliases (`champion`, `candidate`, `previous`), the newest version's state and gate result with every reason, its gate report next to the champion's, the Airflow runs of the latest submission, and links to MLflow, the Ray dashboard and Airflow.

With `--json`: `{"model", "aliases", "newest": {"version", "state", "gate": {"result", "reasons", "report", "champion_report"}}, "airflow": [...], "links": {...}}`.

## `modelport promote NAME VERSION`

Makes `VERSION` the champion and updates serving with the zero-downtime swap, through the `serve_sync` DAG ([canary.md](canary.md)). It asks for confirmation first. It is refused (exit 8) when the version did not pass the gate, and the DAG also refuses when a canary is running for the model or the version is already the champion.

| Option              | Default | Effect                                                                      |
| ------------------- | ------- | --------------------------------------------------------------------------- |
| `--wait`            | off     | Block until serving runs the new champion, then print the aliases.          |
| `--timeout SECONDS` | `1800`  | With `--wait`, give up after this long (exit 12); the run keeps going.      |
| `--yes`             | off     | Do not ask for confirmation.                                                |

## `modelport rollback NAME`

Makes the `previous` version the champion again, with the same swap and the same options as `promote`. Refused (exit 8) when the model has no `previous` version.

With `--json`, `promote` and `rollback` print `{"model", "action", "target", "run_id", "url"}`, plus `"aliases"` with `--wait`.

## `modelport predict NAME FILE`

Sends rows from `FILE` to the model's endpoint (`POST /<model>/predict`, see [serving.md](serving.md)) and prints one prediction per row, with the version and variant (`champion` or `candidate`) that answered. `FILE` is a `.json` file (a request body or a list of rows), a `.csv` or `.parquet` file, or one `.png`/`.jpg` image for a model with exactly one binary input. The rows are checked against the champion's signature before they are sent (exit 4 with the column and a fix); a model with no champion exits 7. One request holds at most 1,024 rows.

| Option     | Default  | Effect                                   |
| ---------- | -------- | ---------------------------------------- |
| `--rows N` | all rows | Send only the first `N` rows (1 to 1,024). |

With `--json`, it prints the `PredictResponse`: `{"model", "version", "variant", "predictions": [{"prediction", "confidence"}]}`.

## Exit codes

| Code | Error                      | Meaning                                                                |
| ---- | -------------------------- | ---------------------------------------------------------------------- |
| `0`  |                            | Success.                                                               |
| `1`  | `ModelportError`           | Any other failure.                                                     |
| `2`  | `SpecError`                | `model.yaml` is missing or invalid, or a name breaks the naming rules. |
| `3`  | `ContractError`            | The model class breaks the `BaseModel` contract.                       |
| `4`  | `SignatureMismatchError`   | Input or output data does not match the signature.                     |
| `5`  | `PackagingError`           | The model folder cannot be packaged.                                   |
| `6`  | `PlatformUnavailableError` | Airflow or MLflow cannot be reached (run `make ui`).                   |
| `7`  | `NotFoundError`            | A model, version, alias or DAG does not exist.                         |
| `8`  | `PromotionRefusedError`    | The gate rejected the version (`submit --wait`).                       |
| `9`  | `IllegalTransitionError`   | A lifecycle change is not allowed from the version's state.            |
| `10` | `RemoteOperationError`     | A pipeline run failed.                                                 |
| `11` | `ConfigError`              | Settings or credentials are missing or invalid.                        |
| `12` | `WaitTimeoutError`         | `--wait` ran out of time.                                              |

## Environment

The CLI reads the environment first, then `.modelport/credentials.env` (written by `make up`, found in the current folder or a parent), then these defaults.

| Variable                      | Default                 | Used for                                                    |
| ----------------------------- | ----------------------- | ----------------------------------------------------------- |
| `MODELPORT_AIRFLOW_URL`       | `http://localhost:8080` | Airflow REST API.                                           |
| `MODELPORT_AIRFLOW_USERNAME`  | `modelport`             | Airflow login.                                              |
| `MODELPORT_AIRFLOW_PASSWORD`  | from `credentials.env`  | Airflow login; required.                                    |
| `MLFLOW_TRACKING_URI`         | `http://localhost:5000` | MLflow tracking and registry (artifacts go through it too). |
| `MODELPORT_RAY_DASHBOARD_URL` | `http://localhost:8265` | Links in `status`.                                          |
| `MODELPORT_SERVE_URL`         | `http://localhost:8000` | Serving (from v0.2).                                        |
| `MODELPORT_HTTP_TIMEOUT`      | `30`                    | Seconds per HTTP request.                                   |
