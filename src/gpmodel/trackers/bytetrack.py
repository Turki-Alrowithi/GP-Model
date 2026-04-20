"""ByteTrack wrapper using `supervision` — stateful, detection-agnostic.

ByteTrack associates detections across frames by IoU and a lightweight
Kalman filter, producing stable track ids. It's the default choice
for short-occlusion scenes (drones, surveillance) and needs no
re-identification model.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from gpmodel.core.types import BBox, Detection, Frame, Track

logger = logging.getLogger(__name__)


class ByteTrackTracker:
    """Wraps `supervision.ByteTrack`.

    Takes our domain `Detection` objects, runs them through ByteTrack,
    and returns our domain `Track` objects with stable ids. Internal
    state (Kalman filters, track buffer) is managed by ByteTrack.
    """

    def __init__(
        self,
        fps: float = 30.0,
        track_activation_threshold: float = 0.25,
        lost_track_buffer: int = 30,
        minimum_matching_threshold: float = 0.8,
        minimum_consecutive_frames: int = 1,
    ) -> None:
        import supervision as sv

        self._sv = sv
        self._tracker = sv.ByteTrack(
            track_activation_threshold=track_activation_threshold,
            lost_track_buffer=lost_track_buffer,
            minimum_matching_threshold=minimum_matching_threshold,
            frame_rate=round(fps),
            minimum_consecutive_frames=minimum_consecutive_frames,
        )
        self._track_first_seen: dict[int, int] = {}
        self._track_last_frame: dict[int, int] = {}

    # ── Public API ──────────────────────────────────────────
    def update(self, detections: list[Detection], frame: Frame) -> list[Track]:
        if not detections:
            return []

        class_name_map = {d.class_id: d.class_name for d in detections}
        sv_dets = self._to_sv_detections(detections)
        tracked = self._tracker.update_with_detections(sv_dets)
        return list(self._from_sv_tracked(tracked, frame.frame_id, class_name_map))

    def reset(self) -> None:
        self._tracker.reset()
        self._track_first_seen.clear()
        self._track_last_frame.clear()

    # ── Conversions ────────────────────────────────────────
    def _to_sv_detections(self, dets: list[Detection]) -> Any:
        xyxy = np.array([d.bbox.as_xyxy() for d in dets], dtype=np.float32)
        confidence = np.array([d.confidence for d in dets], dtype=np.float32)
        class_id = np.array([d.class_id for d in dets], dtype=int)
        return self._sv.Detections(xyxy=xyxy, confidence=confidence, class_id=class_id)

    def _from_sv_tracked(
        self, sv_dets: Any, frame_id: int, class_name_map: dict[int, str]
    ) -> list[Track]:
        if sv_dets.tracker_id is None or len(sv_dets) == 0:
            return []

        tracks: list[Track] = []
        for xyxy, conf, cid, tid in zip(
            sv_dets.xyxy,
            sv_dets.confidence if sv_dets.confidence is not None else [1.0] * len(sv_dets),
            sv_dets.class_id if sv_dets.class_id is not None else [0] * len(sv_dets),
            sv_dets.tracker_id,
            strict=False,
        ):
            track_id = int(tid)
            first_seen = self._track_first_seen.setdefault(track_id, frame_id)
            last_seen = self._track_last_frame.get(track_id, frame_id)
            self._track_last_frame[track_id] = frame_id

            class_id = int(cid)
            tracks.append(
                Track(
                    track_id=track_id,
                    class_id=class_id,
                    class_name=class_name_map.get(class_id, str(class_id)),
                    confidence=float(conf),
                    bbox=BBox(*(float(v) for v in xyxy)),
                    age=frame_id - first_seen + 1,
                    time_since_update=max(0, frame_id - last_seen),
                )
            )
        return tracks
