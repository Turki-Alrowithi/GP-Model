"""Event hierarchy for the Observer/Pub-Sub bus.

Every event is immutable and carries its own timestamp + stream identifier,
so subscribers can filter or route based on source.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from gpmodel.core.types import Detection, Frame, PerfSample, Track


class AlertSeverity(StrEnum):
    """Severity levels matching the downstream backend schema."""

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"
    OPERATIONAL = "OPERATIONAL"


@dataclass(frozen=True, slots=True)
class Event:
    """Base class for every event on the bus."""

    stream_id: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass(frozen=True, slots=True)
class DetectionsReady(Event):
    """Emitted once per frame after detection + (optional) tracking.

    Subscribers that want to overlay or persist raw inference output
    listen for this event.
    """

    frame: Frame | None = None
    detections: tuple[Detection, ...] = ()
    tracks: tuple[Track, ...] = ()


@dataclass(frozen=True, slots=True)
class AlertRaised(Event):
    """Emitted when a rule fires (intruder, weapon, crowd, geofence, ...)."""

    severity: AlertSeverity = AlertSeverity.LOW
    rule_type: str = ""
    title: str = ""
    description: str = ""
    detections: tuple[Detection, ...] = ()
    evidence: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class PerfSampled(Event):
    """Periodic performance snapshot (fps, latency)."""

    sample: PerfSample | None = None


@dataclass(frozen=True, slots=True)
class StreamStateChanged(Event):
    """Emitted when a stream opens, pauses, errors, or closes."""

    state: str = ""  # "opened" | "closed" | "error" | "paused"
    detail: str = ""
