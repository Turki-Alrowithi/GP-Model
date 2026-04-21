"""Weapon rule — high-severity alert on sustained weapon-class detections.

Fine-tuned on a 5-class weapon dataset (Grenade, Knife, Missile, Pistol, Rifle).
Default class whitelist matches the trained vocabulary exactly.

1. filters to a configurable class whitelist,
2. requires a high per-detection confidence,
3. requires the track to have survived `min_consecutive_frames` (so
   a single flickery false-positive frame doesn't fire),
4. cools down per-track to avoid repeated alerts on the same object.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from gpmodel.core.events import AlertRaised, AlertSeverity
from gpmodel.core.types import Detection, Frame, Track
from gpmodel.rules.base import Cooldown, Rule

logger = logging.getLogger(__name__)


@dataclass
class WeaponRule(Rule):
    """Fires a CRITICAL alert when a weapon-class track is sustained.

    Attributes:
        classes: class names to treat as weapons (strings matching the
            detector's vocabulary).
        min_confidence: only consider detections above this threshold.
        min_consecutive_frames: track.age must exceed this, so a
            single flickery false positive can't fire on frame 1.
        severity: alert severity.
        cooldown_s: per-track TTL.
    """

    classes: frozenset[str] = frozenset({"Grenade", "Knife", "Missile", "Pistol", "Rifle"})
    min_confidence: float = 0.55
    min_consecutive_frames: int = 3
    severity: AlertSeverity = AlertSeverity.CRITICAL
    cooldown_s: float = 30.0
    name: str = field(default="weapon_detected", init=False)

    _cooldown: Cooldown = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_cooldown", Cooldown(self.cooldown_s))

    def evaluate(
        self,
        frame: Frame,
        detections: list[Detection],
        tracks: list[Track],
    ) -> list[AlertRaised]:
        if not tracks or not self.classes:
            return []

        alerts: list[AlertRaised] = []
        for track in tracks:
            try:
                if track.class_name not in self.classes:
                    continue
                if track.confidence < self.min_confidence:
                    continue
                if track.age < self.min_consecutive_frames:
                    continue
                if not self._cooldown.allow((self.name, track.track_id)):
                    continue
                alerts.append(self._alert(frame, track))
            except Exception:
                logger.exception("WeaponRule: skipping track %s", track.track_id)
        return alerts

    def _alert(self, frame: Frame, track: Track) -> AlertRaised:
        return AlertRaised(
            stream_id=frame.stream_id,
            severity=self.severity,
            rule_type=self.name,
            title=f"{track.class_name}#{track.track_id} detected (conf={track.confidence:.2f})",
            description=(
                f"Track {track.track_id} classified as '{track.class_name}' with "
                f"confidence {track.confidence:.2f} for {track.age} consecutive frames."
            ),
            evidence={
                "track_id": track.track_id,
                "class_name": track.class_name,
                "confidence": round(track.confidence, 3),
                "age_frames": track.age,
                "frame_id": frame.frame_id,
                "bbox": list(track.bbox.as_xyxy()),
            },
        )
