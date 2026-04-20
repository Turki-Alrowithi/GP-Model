"""Structured logging setup — console (rich) or JSON (structlog)."""

from __future__ import annotations

import logging

import structlog


def configure_logging(level: str = "INFO", fmt: str = "console") -> None:
    """Set up root logging for the application.

    `console` mode uses rich-powered pretty printing for humans;
    `json` mode emits newline-delimited JSON for aggregation.
    """
    log_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=log_level,
        format="%(message)s",
        force=True,
    )

    processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
    ]

    if fmt == "json":
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer())

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )
