"""`modelport`: one command per intention, hiding every platform detail (Facade).

Each command parses its options, calls one service function and renders the result. Errors are printed as
"Error: ... / How to fix: ..." and the process exits with the error's code (section "Exit codes" in the docs).
"""

import json
import os
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any, Final

import click
import typer

from modelport import __version__
from modelport.cli import render
from modelport.clients.airflow import AirflowClient, Orchestrator
from modelport.config import ClientSettings
from modelport.errors import (
    ContractError,
    ModelportError,
    PromotionRefusedError,
    RemoteOperationError,
    SignatureMismatchError,
    SpecError,
)
from modelport.lifecycle import ModelState
from modelport.logs import configure_logging
from modelport.naming import (
    Tag,
    deploy_run_id,
    evaluate_run_id,
    rayjob_name,
    serve_sync_run_id,
    submission_key,
    submit_run_id,
)
from modelport.packaging import build_code_zip
from modelport.registry import ArtifactStore, MlflowRegistry, ModelRegistry
from modelport.spec import spec_sha256
from modelport.validate import validate_model_dir

app = typer.Typer(
    name="modelport",
    help="Put a model into production with one command: train, evaluate, serve and monitor it.",
    no_args_is_help=True,
    add_completion=False,
    pretty_exceptions_enable=False,
)

# Exit code of `validate` by the first failed check (the error class behind that check).
VALIDATE_EXIT_CODES: Final = {
    "dir.exists": SpecError.exit_code,
    "spec.file": SpecError.exit_code,
    "spec.schema": SpecError.exit_code,
    "class.import": ContractError.exit_code,
    "class.subclass": ContractError.exit_code,
    "class.methods": ContractError.exit_code,
    "class.train_loop": ContractError.exit_code,
    "model.load": ContractError.exit_code,
    "model.input_stats": ContractError.exit_code,
    "example.input": SignatureMismatchError.exit_code,
    "model.predict": SignatureMismatchError.exit_code,
}


@dataclass(frozen=True)
class Options:
    """Global options shared by every command."""

    debug: bool
    as_json: bool


@dataclass(frozen=True)
class Services:
    """The adapters the commands use (fakes in tests)."""

    registry: ModelRegistry
    store: ArtifactStore
    orchestrator: Orchestrator
    settings: ClientSettings


def build_services() -> Services:
    """Real adapters from `ClientSettings.from_env()`."""
    settings = ClientSettings.from_env()
    registry = MlflowRegistry(settings.mlflow_url)
    orchestrator = AirflowClient(
        settings.airflow_url,
        settings.airflow_username,
        settings.airflow_password,
        timeout_s=settings.http_timeout_s,
    )
    return Services(registry=registry, store=registry, orchestrator=orchestrator, settings=settings)


TRAIN_DAG: Final = "train_model"
EVALUATE_DAG: Final = "evaluate_and_promote"
DEPLOY_DAG: Final = "deploy_candidate"
SERVE_DAG: Final = "serve_sync"


def _print_version(value: bool) -> None:
    if value:
        typer.echo(f"modelport {__version__}")
        raise typer.Exit


@app.callback()
def main(
    ctx: typer.Context,
    debug: Annotated[
        bool, typer.Option("--debug", help="Show stack traces and HTTP details on errors.")
    ] = False,
    as_json: Annotated[
        bool, typer.Option("--json", help="Print machine-readable JSON instead of tables.")
    ] = False,
    version: Annotated[
        bool,
        typer.Option(
            "--version", callback=_print_version, is_eager=True, help="Print the modelport version and exit."
        ),
    ] = False,
) -> None:
    """Put a model into production with one command: train, evaluate, serve and monitor it."""
    os.environ.setdefault(
        "MLFLOW_SUPPRESS_PRINTING_URL_TO_STDOUT", "true"
    )  # we print our own links; keeps --json clean
    configure_logging(json=as_json, level="DEBUG" if debug else "WARNING")
    ctx.obj = Options(debug=debug, as_json=as_json)


@contextmanager
def handled(ctx: typer.Context) -> Iterator[Options]:
    """Turn errors into the CLI contract: ModelportError -> its exit code; anything else -> exit 1."""
    options: Options = ctx.obj
    try:
        yield options
    except (typer.Exit, click.exceptions.Exit, click.exceptions.Abort, click.ClickException):
        raise
    except ModelportError as err:
        render.error(err, as_json=options.as_json, debug=options.debug)
        raise typer.Exit(err.exit_code) from None
    except Exception as err:
        render.unexpected(err, debug=options.debug)
        raise typer.Exit(1) from None


@app.command()
def validate(
    ctx: typer.Context,
    model_dir: Annotated[
        Path,
        typer.Argument(
            metavar="DIR", help="Folder with model.yaml and your model class.", show_default=False
        ),
    ],
    spec: Annotated[
        str, typer.Option("--spec", metavar="FILE", help="Spec file name inside DIR.")
    ] = "model.yaml",
    no_predict: Annotated[
        bool,
        typer.Option(
            "--no-predict", help="Skip loading the model and running predict on example/input.parquet."
        ),
    ] = False,
) -> None:
    """Check a model folder locally, with no cluster."""
    with handled(ctx) as options:
        report = validate_model_dir(model_dir, spec_file=spec, run_predict=not no_predict)
        render.validation(report, as_json=options.as_json)
        failure = report.first_failure()
        raise typer.Exit(VALIDATE_EXIT_CODES[failure.name] if failure else 0)


@app.command()
def submit(
    ctx: typer.Context,
    model_dir: Annotated[
        Path,
        typer.Argument(
            metavar="DIR", help="Folder with model.yaml and your model class.", show_default=False
        ),
    ],
    spec: Annotated[
        str, typer.Option("--spec", metavar="FILE", help="Spec file name inside DIR.")
    ] = "model.yaml",
    dataset_version: Annotated[
        int | None,
        typer.Option(
            "--dataset-version",
            metavar="N",
            min=1,
            help="Train on dataset version vN instead of data.dataset's version (or the latest).",
        ),
    ] = None,
    wait: Annotated[
        bool, typer.Option("--wait", help="Block until the model is a candidate or rejected.")
    ] = False,
    timeout: Annotated[
        float,
        typer.Option("--timeout", metavar="SECONDS", min=1, help="With --wait, give up after this long."),
    ] = 3600,
) -> None:
    """Validate, package and send a model for training, evaluation and promotion to candidate."""
    with handled(ctx) as options:
        report = validate_model_dir(model_dir, spec_file=spec)
        failure = report.first_failure()
        if failure or report.spec is None:
            render.validation(report, as_json=options.as_json)
            raise typer.Exit(VALIDATE_EXIT_CODES[failure.name] if failure else SpecError.exit_code)
        model_spec, services = report.spec, build_services()
        name, spec_path = model_spec.name, model_dir / spec
        with tempfile.TemporaryDirectory() as tmp:
            package = build_code_zip(model_dir, Path(tmp))
            services.registry.ensure_registered_model(
                name,
                {
                    Tag.DATASET: model_spec.data.dataset_name,
                    Tag.AUTO_RETRAIN: str(model_spec.data.auto_retrain).lower(),
                },
            )
            spec_hash = spec_sha256(spec_path)
            key = submission_key(package.sha256, spec, spec_hash)
            run_id = services.store.find_submission(name, key) or services.store.create_submission(
                name, package, spec_path, spec_hash
            )
        services.registry.set_model_tags(
            name,
            {
                Tag.LATEST_SUBMISSION_RUN_ID: run_id,
                Tag.LATEST_CODE_SHA256: package.sha256,
                Tag.LATEST_SPEC_FILE: spec,
            },
        )
        conf = {
            "model": name,
            "submission_run_id": run_id,
            "dataset_version": dataset_version,
            "spec_file": spec,
        }
        ref = services.orchestrator.trigger(TRAIN_DAG, submit_run_id(name, key), conf)
        result = {
            "model": name,
            "code_sha256": package.sha256,
            "submission_run_id": run_id,
            "train_run_id": ref.run_id,
            "train_url": ref.url,
            "existed": ref.existed,
        }
        if not options.as_json:
            typer.echo(f"Submitted {name} (code {package.sha256[:12]}). Training: {ref.url}")
            if ref.existed:
                typer.echo("This exact code was already submitted; showing the existing run.")
            typer.echo(f"Next: modelport status {name}")
        if wait:
            result["outcome"] = _wait_for_outcome(services, name, ref.run_id, timeout, options)
        if options.as_json:
            render.emit_json(result)


def _wait_for_outcome(
    services: Services, name: str, train_run_id: str, timeout: float, options: Options
) -> str:
    """Wait for training, then the gate; print and return "candidate" (a rejection raises, exit 8)."""
    orchestrator = services.orchestrator
    orchestrator.wait(TRAIN_DAG, train_run_id, timeout_s=timeout)
    if not orchestrator.has_dag(EVALUATE_DAG):
        message = "training finished; evaluation DAG not installed"
        if not options.as_json:
            typer.echo(message)
        return message
    # exact: train_model names its RayJob from the DAG run ID (a retrain reuses the submission run)
    info = services.registry.find_version_by_tag(name, Tag.TRAIN_RAYJOB, rayjob_name(name, train_run_id))
    if info is None:
        raise RemoteOperationError(
            what=f"Training of {name} finished but registered no version (run {train_run_id})",
            fix=f"Run `modelport status {name}` and check the train_model run's logs.",
        )
    orchestrator.wait(EVALUATE_DAG, evaluate_run_id(name, info.version), timeout_s=timeout)
    tags = services.registry.get_version(name, info.version).tags
    if tags.get(Tag.GATE_RESULT) != "passed":
        reasons = "; ".join(json.loads(tags.get(Tag.GATE_REASONS) or "[]")) or "no reason recorded"
        raise PromotionRefusedError(
            what=f"{name} v{info.version} rejected: {reasons}",
            fix=f"Improve the model and submit again; `modelport status {name}` shows the gate report.",
        )
    if not options.as_json:
        typer.echo(f"{name} v{info.version}: candidate")
    return "candidate"


def _gate(services: Services, tags: dict[str, str] | Any) -> dict[str, Any]:  # noqa: ANN401
    """Gate result, reasons and the two reports of one gate run (missing parts are None)."""
    run_id = tags.get(Tag.GATE_EVAL_RUN_ID)
    reports: dict[str, Any] = {"report": None, "champion_report": None}
    for key, path in (("report", "gate/report.json"), ("champion_report", "gate/champion_report.json")):
        if run_id:
            try:
                reports[key] = services.store.read_json(run_id, path)
            except ModelportError:  # a first version has no champion report
                reports[key] = None
    return {
        "result": tags.get(Tag.GATE_RESULT),
        "reasons": json.loads(tags.get(Tag.GATE_REASONS) or "[]"),
        **reports,
    }


def model_status(services: Services, name: str) -> dict[str, Any]:
    """Everything `status` shows, as one JSON-ready dict.

    Raises:
        NotFoundError: no model has this name.
    """
    snapshot = services.registry.snapshot(name)
    newest = services.registry.latest_version(name)
    settings, orchestrator = services.settings, services.orchestrator
    runs: list[tuple[str, str]] = []
    latest_submission = services.registry.model_tags(name).get(Tag.LATEST_SUBMISSION_RUN_ID)
    if latest_submission:
        key = services.store.run_tags(latest_submission).get(Tag.SUBMISSION_KEY)
        if key:
            runs.append((TRAIN_DAG, submit_run_id(name, key)))
    if newest is not None:
        runs += [
            (EVALUATE_DAG, evaluate_run_id(name, newest.version)),
            (DEPLOY_DAG, deploy_run_id(name, newest.version)),
        ]
    airflow = []
    for dag_id, run_id in runs:
        info = orchestrator.get_run(dag_id, run_id)
        airflow.append(
            {
                "dag_id": dag_id,
                "run_id": run_id,
                "state": info.state if info else "not started",
                "url": f"{settings.airflow_url}/dags/{dag_id}/runs/{run_id}",
            }
        )
    return {
        "model": name,
        "aliases": {
            "champion": snapshot.champion,
            "previous": snapshot.previous,
            "candidate": snapshot.candidate,
        },
        "newest": None
        if newest is None
        else {"version": newest.version, "state": str(newest.state), "gate": _gate(services, newest.tags)},
        "airflow": airflow,
        "links": {
            "mlflow": f"{settings.mlflow_url}/#/models/{name}",
            "ray": f"{settings.ray_dashboard_url}/#/jobs",
            "airflow": f"{settings.airflow_url}/dags",
        },
    }


def _render_status(status: dict[str, Any]) -> None:
    aliases = status["aliases"]
    typer.echo(status["model"])
    typer.echo("  " + "   ".join(f"{a} {'v' + v if v else '-'}" for a, v in aliases.items()))
    newest = status["newest"]
    if newest is None:
        typer.echo("  no versions yet")
    else:
        gate = newest["gate"]
        typer.echo(f"  newest v{newest['version']}: {newest['state']}, gate {gate['result'] or 'not run'}")
        for reason in gate["reasons"]:
            typer.echo(f"    - {reason}")
        report, champion = gate["report"], gate["champion_report"]
        if report:
            header = f"  {report['metric_name']:<24}{'v' + report['version']:>10}"
            header += f"{'champion v' + champion['version']:>16}" if champion else ""
            typer.echo(header)

            def row(label: str, ours: float, theirs: float | None, fmt: str = ".3f") -> None:
                line = f"  {label:<24}{ours:>10{fmt}}"
                typer.echo(line + (f"{theirs:>16{fmt}}" if theirs is not None else ""))

            row("metric", report["metric"], champion["metric"] if champion else None)
            for label, value in report["per_class"].items():
                row(f"  {label}", value, champion["per_class"].get(label) if champion else None)
            latency = f"latency ms (batch {report['latency_batch_size']})"
            row(
                latency,
                report["latency_ms_median"],
                champion["latency_ms_median"] if champion else None,
                ".1f",
            )
    for run in status["airflow"]:
        typer.echo(f"  {run['dag_id']:<22}{run['state']:<13}{run['url']}")
    for label, url in status["links"].items():
        typer.echo(f"  {label:<8}{url}")


@app.command()
def status(
    ctx: typer.Context,
    name: Annotated[
        str, typer.Argument(metavar="NAME", help="Model name (from model.yaml).", show_default=False)
    ],
) -> None:
    """Show aliases, the latest version's stage, links to Airflow, Ray and MLflow, and its gate report."""
    with handled(ctx) as options:
        result = model_status(build_services(), name)
        if options.as_json:
            render.emit_json(result)
        else:
            _render_status(result)


def serve_change(
    services: Services, name: str, action: str, version: str | None, *, wait: bool, timeout: float
) -> dict[str, Any]:
    """Check what the change needs, then start `serve_sync` (and wait for it with `wait`).

    Raises:
        PromotionRefusedError: the version did not pass the gate (promote), or there is no previous
            version (rollback).
        NotFoundError: the model or version does not exist.
    """
    snapshot = services.registry.snapshot(name)
    if action == "promote":
        assert version is not None
        info = services.registry.get_version(name, version)
        if info.tags.get(Tag.GATE_RESULT) != "passed" or info.state is ModelState.REJECTED:
            raise PromotionRefusedError(
                what=f"{name} v{version} did not pass the gate",
                fix=f"`modelport status {name}` shows the reasons; promote a version that passed.",
            )
        target = version
    else:
        if snapshot.previous is None:
            raise PromotionRefusedError(
                what=f"{name} has no previous version to roll back to",
                fix=f"Promote a specific version with `modelport promote {name} <version>`.",
            )
        target = snapshot.previous
    run_id = serve_sync_run_id(action, name, datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ"))
    conf = {"model": name, "action": action, "version": version}
    ref = services.orchestrator.trigger(SERVE_DAG, run_id, conf)
    result: dict[str, Any] = {
        "model": name,
        "action": action,
        "target": target,
        "run_id": run_id,
        "url": ref.url,
    }
    if wait:
        services.orchestrator.wait(SERVE_DAG, run_id, timeout_s=timeout)
        after = services.registry.snapshot(name)
        result["aliases"] = {
            "champion": after.champion,
            "previous": after.previous,
            "candidate": after.candidate,
        }
    return result


def _serve_command(
    ctx: typer.Context, name: str, action: str, version: str | None, wait: bool, timeout: float, yes: bool
) -> None:
    with handled(ctx) as options:
        services = build_services()
        if not yes:
            target = f"v{version}" if version else "the previous version"
            typer.confirm(f"Make {target} the champion of {name}?", abort=True, err=True)
        result = serve_change(services, name, action, version, wait=wait, timeout=timeout)
        if options.as_json:
            render.emit_json(result)
            return
        typer.echo(f"{action.capitalize()} {name} to v{result['target']}: {result['url']}")
        if wait:
            aliases = result["aliases"]
            typer.echo(f"  champion v{aliases['champion']}   previous {aliases['previous'] or '-'}")
        else:
            typer.echo(f"Next: modelport status {name}")


WaitOption = Annotated[bool, typer.Option("--wait", help="Block until serving runs the new champion.")]
TimeoutOption = Annotated[
    float, typer.Option("--timeout", metavar="SECONDS", min=1, help="With --wait, give up after this long.")
]
YesOption = Annotated[bool, typer.Option("--yes", help="Do not ask for confirmation.")]
NameArgument = Annotated[
    str, typer.Argument(metavar="NAME", help="Model name (from model.yaml).", show_default=False)
]


@app.command()
def promote(
    ctx: typer.Context,
    name: NameArgument,
    version: Annotated[
        str,
        typer.Argument(metavar="VERSION", help="Version number to make the champion.", show_default=False),
    ],
    wait: WaitOption = False,
    timeout: TimeoutOption = 1800,
    yes: YesOption = False,
) -> None:
    """Make VERSION the champion (only if it passed the gate) and update serving."""
    _serve_command(ctx, name, "promote", version, wait, timeout, yes)


@app.command()
def rollback(
    ctx: typer.Context,
    name: NameArgument,
    wait: WaitOption = False,
    timeout: TimeoutOption = 1800,
    yes: YesOption = False,
) -> None:
    """Make the previous champion the champion again and update serving."""
    _serve_command(ctx, name, "rollback", None, wait, timeout, yes)
