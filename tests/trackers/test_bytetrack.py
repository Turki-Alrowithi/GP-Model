"""Tests for ByteTrackTracker.

Integration-style: we drive real `supervision.ByteTrack` with a
scripted sequence of frames so we can assert track ids are stable
when a detection persists.
"""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pytest

from gpmodel.core.interfaces import Tracker
from gpmodel.core.types import BBox, Detection, Frame
from gpmodel.trackers.bytetrack import ByteTrackTracker


def make_frame(frame_id: int) -> Frame:
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    return Frame(stream_id="cam-1", frame_id=frame_id, timestamp=datetime.now(UTC), image=img)


def det(x1: float, y1: float, x2: float, y2: float, cid: int = 0, conf: float = 0.9) -> Detection:
    return Detection(
        class_id=cid,
        class_name={0: "person", 2: "car"}.get(cid, "other"),
        confidence=conf,
        bbox=BBox(x1, y1, x2, y2),
    )


def test_is_a_tracker() -> None:
    t = ByteTrackTracker(fps=30)
    assert isinstance(t, Tracker)


def test_empty_detections_yields_no_tracks() -> None:
    t = ByteTrackTracker()
    assert t.update([], make_frame(1)) == []


def test_assigns_stable_track_id_across_frames() -> None:
    """A detection that persists should keep the same track id."""
    t = ByteTrackTracker(
        track_activation_threshold=0.25,
        minimum_consecutive_frames=1,
        lost_track_buffer=30,
    )

    # Drive the tracker through 5 frames — the box drifts slightly each frame.
    track_ids: list[int] = []
    for i in range(1, 6):
        dets = [det(100 + i, 100 + i, 200 + i, 200 + i)]
        tracks = t.update(dets, make_frame(i))
        if tracks:
            track_ids.append(tracks[0].track_id)

    # After activation ByteTrack should settle on a single track id.
    assert len(set(track_ids[-3:])) == 1


def test_preserves_class_name_from_detections() -> None:
    t = ByteTrackTracker(minimum_consecutive_frames=1)

    for i in range(1, 6):
        tracks = t.update([det(100, 100, 200, 200, cid=2)], make_frame(i))
    assert tracks
    assert tracks[0].class_id == 2
    assert tracks[0].class_name == "car"


def test_reset_clears_internal_state() -> None:
    t = ByteTrackTracker(minimum_consecutive_frames=1)
    for i in range(1, 6):
        t.update([det(100, 100, 200, 200)], make_frame(i))

    t.reset()
    # After reset a fresh activation must start again; ages restart at 1.
    for i in range(1, 4):
        tracks = t.update([det(300, 300, 400, 400)], make_frame(i))
    assert tracks
    assert tracks[0].age <= 3


@pytest.mark.parametrize("fps", [15, 30, 60])
def test_accepts_different_frame_rates(fps: int) -> None:
    t = ByteTrackTracker(fps=fps)
    assert t._tracker is not None  # smoke check
