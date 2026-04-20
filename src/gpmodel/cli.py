"""CLI entrypoint — `gpmodel` / `python -m gpmodel` / apps/run_inference.py.

Wires everything together from a YAML config:
config → components (Factory) → engine → run → print summary.
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
from pathlib import Path
from types import FrameType

from rich.console import Console

from gpmodel.config import build_engine, load_config
from gpmodel.core.dispatcher import AlertDispatcher
from gpmodel.telemetry.logging import configure_logging

logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="gpmodel",
        description="GP-Model — real-time security drone detection engine.",
    )
    parser.add_argument(
        "--config",
        "-c",
        type=Path,
        default=Path("configs/laptop.yaml"),
        help="Path to YAML config (default: configs/laptop.yaml)",
    )
    args = parser.parse_args(argv)

    config = load_config(args.config)
    configure_logging(level=config.logging.level, fmt=config.logging.format)
    logger.info("Loaded config from %s", args.config)

    dispatcher = AlertDispatcher()
    engine, metrics_sub = build_engine(config, dispatcher)

    # Graceful shutdown on Ctrl-C / SIGTERM
    def _handle_signal(signum: int, _frame: FrameType | None) -> None:
        logger.info("Received signal %d — stopping", signum)
        engine.stop()

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    try:
        engine.run()
    except KeyboardInterrupt:
        pass
    finally:
        if metrics_sub is not None:
            _print_summary(metrics_sub.summary())

    return 0


def _print_summary(summary: object) -> None:
    from gpmodel.publishers.metrics import MetricsSummary

    if not isinstance(summary, MetricsSummary):
        return
    console = Console()
    console.rule("[bold]Run summary[/bold]")
    console.print(f"Frames processed:  {summary.frames}")
    console.print(f"Perf samples:      {summary.perf_samples}")
    console.print(f"Avg FPS:           {summary.avg_fps:.1f}")
    console.print(f"Avg latency:       {summary.avg_latency_ms:.1f} ms/frame")
    console.print(f"Alerts total:      {summary.alerts_total}")
    if summary.alerts_by_severity:
        console.print(f"  by severity:     {dict(summary.alerts_by_severity)}")
    if summary.alerts_by_rule:
        console.print(f"  by rule:         {dict(summary.alerts_by_rule)}")


if __name__ == "__main__":
    sys.exit(main())
