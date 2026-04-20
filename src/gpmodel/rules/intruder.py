"""Intruder rule — alert when a person track isn't recognised as staff.

For each person track we crop the bounding-box region from the frame,
run the face encoder on the crop, and compare any detected face to the
enrolled staff embeddings. The classification is cached per track_id
so we don't re-identify the same person every frame.

Classification states per track:

- UNKNOWN      — no classification yet (initial state).
- INDETERMINATE — face encoder found no face in the crop; retry later.
- STAFF        — matched a staff embedding; never re-check.
- INTRUDER     — face seen but no staff match; alert (cooldown'd).

The rule deliberately does not alert on "indeterminate" — absence of
evidence isn't evidence of intrusion. Phase 3 will add fall-back body
ReID (OSNet) so we can still identify people whose faces aren't
visible from drone altitude.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np

from gpmodel.core.events import AlertRaised, AlertSeverity
from gpmodel.core.types import Detection, Frame, Track
from gpmodel.reid.face_db import StaffFaceDB
from gpmodel.rules.base import Cooldown, Rule

logger = logging.getLogger(__name__)


class _State(StrEnum):
    UNKNOWN = "unknown"
    INDETERMINATE = "indeterminate"
    STAFF = "staff"
    INTRUDER = "intruder"


@dataclass
class IntruderRule(Rule):
    """Flag person tracks that don't match any staff embedding.

    Attributes:
        staff_db: enrolled staff face DB.
        classes: detector class names that should trigger identity checks
            (default persons).
        min_consecutive_frames: don't attempt identification on
            one-frame tracks; wait for ByteTrack to confirm.
        indeterminate_retry_every: frames between retries when the last
            identification was INDETERMINATE (no face visible).
        bbox_padding: percent padding added to the person bbox before
            cropping — tiny bboxes often clip the head off.
        severity: alert severity.
        cooldown_s: per-track cooldown on the INTRUDER alert.
    """

    staff_db: StaffFaceDB
    classes: frozenset[str] = frozenset({"person"})
    min_consecutive_frames: int = 2
    indeterminate_retry_every: int = 15
    bbox_padding: float = 0.10
    severity: AlertSeverity = AlertSeverity.HIGH
    cooldown_s: float = 60.0
    name: str = field(default="intruder_detected", init=False)

    _state: dict[int, _State] = field(default_factory=dict, init=False)
    _last_try: dict[int, int] = field(default_factory=dict, init=False)
    _staff_name: dict[int, str] = field(default_factory=dict, init=False)
    _cooldown: Cooldown = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_cooldown", Cooldown(self.cooldown_s))

    # ── Rule API ───────────────────────────────────────────
    def evaluate(
        self,
        frame: Frame,
        detections: list[Detection],
        tracks: list[Track],
    ) -> list[AlertRaised]:
        if not tracks:
            return []

        alerts: list[AlertRaised] = []
        for track in tracks:
            if track.class_name not in self.classes:
                continue
            if track.age < self.min_consecutive_frames:
                continue

            state = self._classify_if_needed(frame, track)
            if state != _State.INTRUDER:
                continue
            if not self._cooldown.allow((self.name, track.track_id)):
                continue
            alerts.append(self._alert(frame, track))
        return alerts

    # ── Classification ────────────────────────────────────
    def _classify_if_needed(self, frame: Frame, track: Track) -> _State:
        state = self._state.get(track.track_id, _State.UNKNOWN)

        # Terminal states — don't redo work.
        if state in (_State.STAFF, _State.INTRUDER):
            return state

        # Indeterminate state: retry every N frames; not every frame.
        if state == _State.INDETERMINATE:
            last = self._last_try.get(track.track_id, 0)
            if frame.frame_id - last < self.indeterminate_retry_every:
                return state

        new_state = self._run_identification(frame, track)
        self._state[track.track_id] = new_state
        self._last_try[track.track_id] = frame.frame_id
        return new_state

    def _run_identification(self, frame: Frame, track: Track) -> _State:
        crop = self._crop(frame, track)
        if crop.size == 0:
            return _State.INDETERMINATE
        faces = self.staff_db.encoder.encode(crop)
        if not faces:
            return _State.INDETERMINATE
        best_face = max(faces, key=lambda f: f.det_score)
        match = self.staff_db.match(best_face.embedding)
        if match is not None:
            self._staff_name[track.track_id] = match.name
            logger.info(
                "Track %d identified as staff '%s' (sim=%.2f)",
                track.track_id,
                match.name,
                match.similarity,
            )
            return _State.STAFF
        return _State.INTRUDER

    def _crop(self, frame: Frame, track: Track) -> np.ndarray:
        h, w = frame.image.shape[:2]
        x1, y1, x2, y2 = track.bbox.as_xyxy()
        pad_x = (x2 - x1) * self.bbox_padding
        pad_y = (y2 - y1) * self.bbox_padding
        cx1 = max(0, int(x1 - pad_x))
        cy1 = max(0, int(y1 - pad_y))
        cx2 = min(w, int(x2 + pad_x))
        cy2 = min(h, int(y2 + pad_y))
        if cx2 <= cx1 or cy2 <= cy1:
            return np.zeros((0, 0, 3), dtype=np.uint8)
        return frame.image[cy1:cy2, cx1:cx2]

    # ── Alert ─────────────────────────────────────────────
    def _alert(self, frame: Frame, track: Track) -> AlertRaised:
        return AlertRaised(
            stream_id=frame.stream_id,
            severity=self.severity,
            rule_type=self.name,
            title=f"Unrecognised person (track #{track.track_id})",
            description=(
                f"Track {track.track_id} (person, conf={track.confidence:.2f}) has "
                f"a visible face that doesn't match any enrolled staff member."
            ),
            evidence={
                "track_id": track.track_id,
                "confidence": round(track.confidence, 3),
                "bbox": list(track.bbox.as_xyxy()),
                "frame_id": frame.frame_id,
                "staff_db_size": self.staff_db.size,
            },
        )
