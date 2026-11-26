"""`modelport`: one command per intention, hiding every platform detail (Facade).

Each command parses its options, calls one service function and renders the result. Errors are printed as
"Error: ... / How to fix: ..." and the process exits with the error's code (section "Exit codes" in the docs).
"""

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Final

import click
import typer

from modelport import __version__
from modelport.cli import render
from modelport.errors import ContractError, ModelportError, SignatureMismatchError, SpecError
from modelport.logs import configure_logging
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
