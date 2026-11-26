"""Structured logging: JSON lines for platform processes, readable console output for the CLI."""

import logging
import sys

import structlog


def configure_logging(*, json: bool, level: str = "INFO") -> None:
    """Configure structlog once per process.

    Args:
        json: one JSON object per line (`timestamp`, `level`, `event`, plus bound context) when True;
            human console output when False.
        level: minimum level name, e.g. "INFO" or "DEBUG".
    """
    renderer: structlog.typing.Processor = (
        structlog.processors.JSONRenderer() if json else structlog.dev.ConsoleRenderer(colors=False)
    )
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.getLevelNamesMapping()[level.upper()]),
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
        cache_logger_on_first_use=False,
    )


def get_logger(name: str) -> structlog.typing.FilteringBoundLogger:
    """A logger whose records carry `logger=<name>`."""
    logger: structlog.typing.FilteringBoundLogger = structlog.get_logger().bind(logger=name)
    return logger
