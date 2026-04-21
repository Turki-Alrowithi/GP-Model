"""Geofence rule — fire an alert when a tracked object enters a restricted zone.

The rule takes a list of named polygons. On every frame each track
that matches `classes` is tested by a *foot-point* (bottom-center of
its bounding box) — the real-world spot where a person stands on the
ground — against every polygon.

Polygons are defined in normalized image coordinates (0-1) by default
so the same config survives resolution changes.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field

from shapely.geometry import Point, Polygon

from gpmodel.core.events import AlertRaised, AlertSeverity
from gpmodel.core.types import Detection, Frame, Track
from gpmodel.rules.base import Cooldown, Rule

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class Geofence:
    """A named restricted polygon in normalized (0-1) or pixel coordinates."""

    name: str
    points: tuple[tuple[float, float], ...]
    normalized: bool = True

    def to_shapely(self, width: int, height: int) -> Polygon:
        if self.normalized:
            scaled = [(x * width, y * height) for x, y in self.points]
        else:
            scaled = [(x, y) for x, y in self.points]
        return Polygon(scaled)


@dataclass
class GeofenceRule(Rule):
    """Fires HIGH-severity alerts when a tracked object enters any fence.

    Attributes:
        zones: list of Geofence polygons.
        classes: class names to test; empty → all classes.
        severity: alert severity (default HIGH).
        cooldown_s: per-track TTL so a loitering person doesn't flood alerts.
        foot_point: use bottom-center instead of bbox center for the test.
    """

    zones: Sequence[Geofence]
    classes: frozenset[str] = frozenset()
    severity: AlertSeverity = AlertSeverity.HIGH
    cooldown_s: float = 30.0
    foot_point: bool = True
    name: str = field(default="geofence_breach", init=False)

    _cooldown: Cooldown = field(init=False)
    _compiled: dict[str, Polygon] = field(default_factory=dict, init=False)
    _compiled_for: tuple[int, int] | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_cooldown", Cooldown(self.cooldown_s))

    # ── Rule API ───────────────────────────────────────────
    def evaluate(
        self,
        frame: Frame,
        detections: list[Detection],
        tracks: list[Track],
    ) -> list[AlertRaised]:
        if not tracks or not self.zones:
            return []

        polys = self._get_compiled(frame.width, frame.height)
        alerts: list[AlertRaised] = []

        for track in tracks:
            try:
                if self.classes and track.class_name not in self.classes:
                    continue
                px, py = self._probe_point(track)
                point = Point(px, py)
                for name, poly in polys.items():
                    if not poly.contains(point):
                        continue
                    if not self._cooldown.allow((self.name, track.track_id, name)):
                        continue
                    alerts.append(self._alert(frame, track, name, px, py))
            except Exception:
                logger.exception("GeofenceRule: skipping track %s", track.track_id)
        return alerts

    # ── Internals ──────────────────────────────────────────
    def _probe_point(self, track: Track) -> tuple[float, float]:
        if self.foot_point:
            cx = (track.bbox.x1 + track.bbox.x2) / 2.0
            return cx, track.bbox.y2
        return track.bbox.center

    def _get_compiled(self, width: int, height: int) -> dict[str, Polygon]:
        if self._compiled_for != (width, height):
            self._compiled = {z.name: z.to_shapely(width, height) for z in self.zones}
            object.__setattr__(self, "_compiled_for", (width, height))
        return self._compiled

    def _alert(self, frame: Frame, track: Track, zone: str, px: float, py: float) -> AlertRaised:
        return AlertRaised(
            stream_id=frame.stream_id,
            severity=self.severity,
            rule_type=self.name,
            title=f"{track.class_name}#{track.track_id} entered zone '{zone}'",
            description=(
                f"Track {track.track_id} ({track.class_name}, conf={track.confidence:.2f}) "
                f"at ({px:.0f},{py:.0f}) is inside restricted zone '{zone}'."
            ),
            evidence={
                "zone": zone,
                "track_id": track.track_id,
                "class_name": track.class_name,
                "confidence": round(track.confidence, 3),
                "foot_point": [round(px, 1), round(py, 1)],
                "frame_id": frame.frame_id,
            },
        )
