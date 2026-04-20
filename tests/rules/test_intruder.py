"""Tests for IntruderRule — uses a stub FaceEncoder and a real StaffFaceDB."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from gpmodel.core.events import AlertSeverity
from gpmodel.core.types import BBox, Frame, Track
from gpmodel.reid.encoder import FaceEmbedding
from gpmodel.reid.face_db import StaffFaceDB
from gpmodel.rules.intruder import IntruderRule


@dataclass
class StubEncoder:
    queue: list[list[FaceEmbedding]] = field(default_factory=list)

    def encode(self, image: np.ndarray) -> list[FaceEmbedding]:
        return self.queue.pop(0) if self.queue else []


def _emb(vec: list[float], score: float = 0.99) -> FaceEmbedding:
    v = np.array(vec, dtype=np.float32)
    v /= np.linalg.norm(v) + 1e-9
    return FaceEmbedding(bbox=(0.0, 0.0, 10.0, 10.0), embedding=v, det_score=score)


def _track(tid: int, cls: str = "person", age: int = 5) -> Track:
    return Track(
        track_id=tid,
        class_id=0,
        class_name=cls,
        confidence=0.9,
        bbox=BBox(100, 100, 400, 700),
        age=age,
        time_since_update=0,
    )


def _staff_db_with(
    alice_vec: list[float], encoder: StubEncoder, threshold: float = 0.5
) -> StaffFaceDB:
    db = StaffFaceDB(encoder=encoder, threshold=threshold)
    # Directly seed embeddings so we don't need filesystem fixtures here.
    db._embeddings["alice"] = [np.array(alice_vec, dtype=np.float32) / np.linalg.norm(alice_vec)]
    return db


def test_intruder_fires_when_face_does_not_match(sample_frame: Frame) -> None:
    enc = StubEncoder(queue=[[_emb([0, 0, 1])]])  # probe face — very different
    db = _staff_db_with([1, 0, 0], enc)
    rule = IntruderRule(staff_db=db, min_consecutive_frames=1)

    alerts = rule.evaluate(sample_frame, [], [_track(1)])
    assert len(alerts) == 1
    assert alerts[0].rule_type == "intruder_detected"
    assert alerts[0].severity == AlertSeverity.HIGH
    assert alerts[0].evidence["track_id"] == 1


def test_staff_match_does_not_fire(sample_frame: Frame) -> None:
    enc = StubEncoder(queue=[[_emb([0.95, 0.05, 0])]])  # near-match to alice
    db = _staff_db_with([1, 0, 0], enc)
    rule = IntruderRule(staff_db=db, min_consecutive_frames=1)

    assert rule.evaluate(sample_frame, [], [_track(1)]) == []


def test_indeterminate_does_not_fire(sample_frame: Frame) -> None:
    # No face found — rule stays silent, doesn't guess.
    enc = StubEncoder(queue=[[]])
    db = _staff_db_with([1, 0, 0], enc)
    rule = IntruderRule(staff_db=db, min_consecutive_frames=1)

    assert rule.evaluate(sample_frame, [], [_track(1)]) == []


def test_classification_cached_for_staff(sample_frame: Frame) -> None:
    # Encoder responds on the first call; if it's invoked again the
    # queue would be empty and we'd get [] (indeterminate).
    enc = StubEncoder(queue=[[_emb([0.98, 0.02, 0])]])
    db = _staff_db_with([1, 0, 0], enc)
    rule = IntruderRule(staff_db=db, min_consecutive_frames=1)

    # First frame classifies and caches as staff.
    rule.evaluate(sample_frame, [], [_track(1)])
    # Second frame should not re-encode.
    rule.evaluate(sample_frame, [], [_track(1)])
    assert enc.queue == []  # only consumed once


def test_non_person_class_is_ignored(sample_frame: Frame) -> None:
    enc = StubEncoder(queue=[[_emb([0, 0, 1])]])
    db = _staff_db_with([1, 0, 0], enc)
    rule = IntruderRule(staff_db=db, min_consecutive_frames=1)

    # car → skipped before encoder is even called.
    assert rule.evaluate(sample_frame, [], [_track(1, cls="car")]) == []
    # Encoder was not called — its scripted response is still queued.
    assert len(enc.queue) == 1


def test_young_tracks_are_skipped(sample_frame: Frame) -> None:
    enc = StubEncoder(queue=[[_emb([0, 0, 1])]])
    db = _staff_db_with([1, 0, 0], enc)
    rule = IntruderRule(staff_db=db, min_consecutive_frames=5)

    assert rule.evaluate(sample_frame, [], [_track(1, age=2)]) == []
    # Encoder not called either.
    assert len(enc.queue) == 1


def test_intruder_cooldown_suppresses_refire(sample_frame: Frame) -> None:
    enc = StubEncoder(queue=[[_emb([0, 0, 1])]])
    db = _staff_db_with([1, 0, 0], enc)
    rule = IntruderRule(staff_db=db, min_consecutive_frames=1, cooldown_s=60.0)

    first = rule.evaluate(sample_frame, [], [_track(1)])
    second = rule.evaluate(sample_frame, [], [_track(1)])
    assert len(first) == 1
    assert second == []
