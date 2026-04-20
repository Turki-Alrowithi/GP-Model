"""Crowd rule — alert when N+ tracked targets stay in an area for T seconds.

Implementation is deliberately the simple "box count in polygon"
variant; density-map models (CSRNet, MCNN) are Phase 3 once we have
drone footage dense enough to justify them.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime

from shapely.geometry import Point, Polygon

from gpmodel.core.events import AlertRaised, AlertSeverity
from gpmodel.core.types import Detection, Frame, Track
from gpmodel.rules.base import Cooldown, Rule
from gpmodel.rules.geofence import Geofence


@dataclass
class CrowdRule(Rule):
    """Fires an alert when `threshold` tracks persist in `zone` for `min_duration_s`.

    Attributes:
        threshold: minimum number of tracks to count as a crowd.
        zone: optional ROI polygon; if None, the whole frame is used.
        classes: class names to include in the count (default persons).
        min_duration_s: how long the crowd has to persist before alerting.
        severity: alert severity.
        cooldown_s: per-zone TTL after firing so we don't spam.
    """

    threshold: int
    zone: Geofence | None = None
    classes: frozenset[str] = frozenset({"person"})
    min_duration_s: float = 3.0
    severity: AlertSeverity = AlertSeverity.MEDIUM
    cooldown_s: float = 60.0
    name: str = field(default="crowd_formed", init=False)

    _cooldown: Cooldown = field(init=False)
    _persisting_since: datetime | None = field(default=None, init=False)
    _compiled: Polygon | None = field(default=None, init=False)
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
        poly = self._get_zone(frame.width, frame.height)
        relevant = self._filter(tracks, poly)
        count = len(relevant)

        # Stateful "sustained condition" — reset if count drops below threshold.
        now = frame.timestamp.astimezone(UTC)
        if count < self.threshold:
            self._persisting_since = None
            return []

        if self._persisting_since is None:
            self._persisting_since = now
            return []

        elapsed = (now - self._persisting_since).total_seconds()
        if elapsed < self.min_duration_s:
            return []

        zone_name = self.zone.name if self.zone else "frame"
        if not self._cooldown.allow((self.name, zone_name)):
            return []

        return [self._alert(frame, relevant, zone_name, elapsed)]

    # ── Internals ──────────────────────────────────────────
    def _get_zone(self, width: int, height: int) -> Polygon | None:
        if self.zone is None:
            return None
        if self._compiled_for != (width, height):
            object.__setattr__(self, "_compiled", self.zone.to_shapely(width, height))
            object.__setattr__(self, "_compiled_for", (width, height))
        return self._compiled

    def _filter(self, tracks: Sequence[Track], poly: Polygon | None) -> list[Track]:
        out: list[Track] = []
        for t in tracks:
            if self.classes and t.class_name not in self.classes:
                continue
            if poly is not None:
                cx = (t.bbox.x1 + t.bbox.x2) / 2.0
                if not poly.contains(Point(cx, t.bbox.y2)):
                    continue
            out.append(t)
        return out

    def _alert(
        self,
        frame: Frame,
        relevant: list[Track],
        zone_name: str,
        elapsed_s: float,
    ) -> AlertRaised:
        return AlertRaised(
            stream_id=frame.stream_id,
            severity=self.severity,
            rule_type=self.name,
            title=f"Crowd of {len(relevant)} in '{zone_name}' for {elapsed_s:.1f}s",
            description=(
                f"{len(relevant)} {'/'.join(sorted(self.classes)) or 'targets'} "
                f"have been inside '{zone_name}' for at least {self.min_duration_s:.1f}s."
            ),
            evidence={
                "zone": zone_name,
                "count": len(relevant),
                "threshold": self.threshold,
                "elapsed_s": round(elapsed_s, 2),
                "track_ids": [t.track_id for t in relevant],
                "frame_id": frame.frame_id,
            },
        )
