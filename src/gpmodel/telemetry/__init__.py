"""Telemetry utilities — performance metering and structured logging."""

from gpmodel.telemetry.logging import configure_logging
from gpmodel.telemetry.perf import PerfMeter

__all__ = ["PerfMeter", "configure_logging"]
