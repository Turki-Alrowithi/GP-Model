"""Rules engine — evaluates domain rules against detections/tracks each frame.

Each Rule is a self-contained unit that looks at the current frame's
detections/tracks and optionally fires AlertRaised events. The
RulesEngine just fans out to every registered rule; ordering and
dependencies between rules are deliberately out of scope — rules
should be independent.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from datetime import UTC, datetime

from gpmodel.core.events import AlertRaised
from gpmodel.core.types import Detection, Frame, Track


class Rule(ABC):
    """Abstract base for a single detection rule."""

    name: str  # short identifier, used by Cooldown

    @abstractmethod
    def evaluate(
        self,
        frame: Frame,
        detections: list[Detection],
        tracks: list[Track],
    ) -> list[AlertRaised]:
        """Inspect the current frame and emit zero or more alerts."""


class Cooldown:
    """Per-key TTL so one track doesn't emit the same alert every frame.

    Example keys: `("geofence", track_id=42)` or
    `("crowd", zone="main-plaza")`.
    """

    def __init__(self, seconds: float = 30.0) -> None:
        self._seconds = seconds
        self._last_fired: dict[tuple[object, ...], datetime] = {}

    def allow(self, key: tuple[object, ...], now: datetime | None = None) -> bool:
        """Return True if enough time has passed since `key` last fired."""
        at = now or datetime.now(UTC)
        last = self._last_fired.get(key)
        if last is None or (at - last).total_seconds() >= self._seconds:
            self._last_fired[key] = at
            return True
        return False

    def reset(self, key: tuple[object, ...] | None = None) -> None:
        if key is None:
            self._last_fired.clear()
        else:
            self._last_fired.pop(key, None)


class RulesEngine:
    """Holds a list of rules and evaluates them in order per frame."""

    def __init__(self, rules: Iterable[Rule] | None = None) -> None:
        self._rules: list[Rule] = list(rules) if rules else []

    def add(self, rule: Rule) -> None:
        self._rules.append(rule)

    def rules(self) -> list[Rule]:
        return list(self._rules)

    def evaluate(
        self,
        frame: Frame,
        detections: list[Detection],
        tracks: list[Track],
    ) -> list[AlertRaised]:
        alerts: list[AlertRaised] = []
        for rule in self._rules:
            alerts.extend(rule.evaluate(frame, detections, tracks))
        return alerts
