"""Tests for StaffFaceDB — use a fake encoder so tests are offline + fast."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import pytest

from gpmodel.reid.encoder import FaceEmbedding
from gpmodel.reid.face_db import StaffFaceDB


@dataclass
class StubEncoder:
    """Fake FaceEncoder.

    On each call returns whatever is in `responses` keyed by an
    increasing counter, so tests can script a sequence.
    """

    responses: list[list[FaceEmbedding]] = field(default_factory=list)
    calls: list[np.ndarray] = field(default_factory=list)

    def encode(self, image: np.ndarray) -> list[FaceEmbedding]:
        self.calls.append(image)
        if not self.responses:
            return []
        return self.responses.pop(0)


def _emb(vec: list[float], score: float = 0.99) -> FaceEmbedding:
    v = np.array(vec, dtype=np.float32)
    v /= np.linalg.norm(v) + 1e-9
    return FaceEmbedding(bbox=(0.0, 0.0, 10.0, 10.0), embedding=v, det_score=score)


def _dummy_image(tmp_path: Path, name: str) -> Path:
    img = np.zeros((20, 20, 3), dtype=np.uint8)
    out = tmp_path / name
    cv2.imwrite(str(out), img)
    return out


def test_enroll_directory_counts_embeddings(tmp_path: Path) -> None:
    alice = tmp_path / "alice"
    alice.mkdir()
    _dummy_image(alice, "01.jpg")
    _dummy_image(alice, "02.jpg")
    bob = tmp_path / "bob"
    bob.mkdir()
    _dummy_image(bob, "portrait.png")

    encoder = StubEncoder(
        responses=[
            [_emb([1, 0, 0])],
            [_emb([1, 0, 0])],
            [_emb([0, 1, 0])],
        ]
    )
    db = StaffFaceDB(encoder=encoder)

    added = db.enroll_directory(tmp_path)
    assert added == 3
    assert db.size == 3
    assert db.names == ["alice", "bob"]


def test_enroll_skips_image_without_face(tmp_path: Path) -> None:
    alice = tmp_path / "alice"
    alice.mkdir()
    _dummy_image(alice, "blurry.jpg")

    encoder = StubEncoder(responses=[[]])  # no face detected
    db = StaffFaceDB(encoder=encoder)
    assert db.enroll_directory(tmp_path) == 0
    assert db.size == 0


def test_enroll_nonexistent_directory_is_zero(tmp_path: Path) -> None:
    db = StaffFaceDB(encoder=StubEncoder())
    assert db.enroll_directory(tmp_path / "nope") == 0


def test_match_returns_staff_above_threshold(tmp_path: Path) -> None:
    alice = tmp_path / "alice"
    alice.mkdir()
    _dummy_image(alice, "01.jpg")
    encoder = StubEncoder(responses=[[_emb([1, 0, 0])]])
    db = StaffFaceDB(encoder=encoder, threshold=0.5)
    db.enroll_directory(tmp_path)

    probe = np.array([0.95, 0.05, 0.0], dtype=np.float32)
    result = db.match(probe)
    assert result is not None
    assert result.name == "alice"
    assert result.similarity > 0.5


def test_match_returns_none_below_threshold(tmp_path: Path) -> None:
    alice = tmp_path / "alice"
    alice.mkdir()
    _dummy_image(alice, "01.jpg")
    encoder = StubEncoder(responses=[[_emb([1, 0, 0])]])
    db = StaffFaceDB(encoder=encoder, threshold=0.9)
    db.enroll_directory(tmp_path)

    # Very different embedding.
    probe = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    assert db.match(probe) is None


def test_match_returns_none_when_db_empty() -> None:
    db = StaffFaceDB(encoder=StubEncoder())
    probe = np.array([1, 0, 0], dtype=np.float32)
    assert db.match(probe) is None


# Ensure pytest is imported for conftest discovery paths.
_ = pytest
