"""Human and JSON output for the CLI. Command functions call these; no business logic lives here."""

import json
import traceback
from collections.abc import Mapping
from typing import Any

import typer

from modelport.errors import ModelportError
from modelport.validate import CheckStatus, ValidationReport

SYMBOLS = {
    CheckStatus.PASSED: "✔",
    CheckStatus.FAILED: "✖",
    CheckStatus.WARNING: "!",
    CheckStatus.SKIPPED: "-",
}


def emit_json(payload: Mapping[str, Any]) -> None:
    """One JSON document on stdout."""
    typer.echo(json.dumps(payload, indent=2, default=str))


def validation(report: ValidationReport, *, as_json: bool) -> None:
    """The check table, then what failed first and how to fix it."""
    failure = report.first_failure()
    if as_json:
        emit_json(
            {
                "model_dir": str(report.model_dir),
                "ok": report.ok,
                "checks": [c.__dict__ for c in report.checks],
                "first_failure": failure.__dict__ if failure else None,
            }
        )
        return
    width = max(len(c.name) for c in report.checks)
    for check in report.checks:
        typer.echo(f"{SYMBOLS[check.status]} {check.name.ljust(width)}  {check.message}")
    for warning in (c for c in report.checks if c.status is CheckStatus.WARNING):
        typer.echo(f"\nWarning: {warning.message}\n  {warning.fix}")
    if failure:
        typer.echo(f"\nError: {failure.message}\nHow to fix: {failure.fix}", err=False)
    else:
        typer.echo(f"\n{report.model_dir} is ready to submit.")


def error(err: ModelportError, *, as_json: bool, debug: bool) -> None:
    """`Error: <what>` and `How to fix: <fix>`; details only with --json or --debug."""
    if as_json:
        emit_json(
            {
                "error": {
                    "type": type(err).__name__,
                    "what": err.what,
                    "fix": err.fix,
                    "exit_code": err.exit_code,
                    "details": dict(err.details),
                }
            }
        )
        return
    typer.echo(f"Error: {err.what}\nHow to fix: {err.fix}", err=True)
    if debug:
        if err.details:
            typer.echo(f"Details: {json.dumps(dict(err.details), default=str)}", err=True)
        typer.echo("".join(traceback.format_exception(err)), err=True)


def unexpected(err: BaseException, *, debug: bool) -> None:
    """An error modelport did not anticipate: a one-line summary, and the traceback only with --debug."""
    typer.echo(f"Unexpected error: {type(err).__name__}: {err}", err=True)
    typer.echo("How to fix: re-run with --debug and report the output.", err=True)
    if debug:
        typer.echo("".join(traceback.format_exception(err)), err=True)
