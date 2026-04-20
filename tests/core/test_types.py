"""Tests for core domain types."""

from __future__ import annotations

import dataclasses
from datetime import UTC, datetime

import numpy as np
import pytest

from gpmodel.core.types import BBox, Detection, Frame, Track


class TestBBox:
    def test_geometry(self) -> None:
        box = BBox(10, 20, 110, 220)
        assert box.width == 100
        assert box.height == 200
        assert box.area == 20_000
        assert box.center == (60.0, 120.0)
        assert box.as_xyxy() == (10, 20, 110, 220)
        assert box.as_xywh() == (10, 20, 100, 200)

    def test_iou_identical_boxes(self) -> None:
        box = BBox(0, 0, 10, 10)
        assert box.iou(box) == pytest.approx(1.0)

    def test_iou_disjoint_boxes(self) -> None:
        assert BBox(0, 0, 10, 10).iou(BBox(20, 20, 30, 30)) == 0.0

    def test_iou_partial_overlap(self) -> None:
        # 10x10 each, overlap 5x5 = 25; union = 100 + 100 - 25 = 175; iou = 25/175
        iou = BBox(0, 0, 10, 10).iou(BBox(5, 5, 15, 15))
        assert iou == pytest.approx(25 / 175)

    def test_rejects_inverted_coordinates(self) -> None:
        with pytest.raises(ValueError):
            BBox(10, 10, 5, 20)
        with pytest.raises(ValueError):
            BBox(10, 10, 20, 5)

    def test_is_immutable(self) -> None:
        box = BBox(0, 0, 10, 10)
        with pytest.raises(dataclasses.FrozenInstanceError):
            box.x1 = 99  # type: ignore[misc]


class TestDetection:
    def test_defaults(self) -> None:
        d = Detection(class_id=0, class_name="person", confidence=0.9, bbox=BBox(0, 0, 1, 1))
        assert d.track_id is None
        assert d.extra == {}


class TestTrack:
    def test_construction(self) -> None:
        t = Track(
            track_id=1,
            class_id=0,
            class_name="person",
            confidence=0.85,
            bbox=BBox(0, 0, 1, 1),
            age=5,
            time_since_update=0,
        )
        assert t.track_id == 1


class TestFrame:
    def test_derived_geometry(self) -> None:
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        f = Frame(
            stream_id="cam-1",
            frame_id=0,
            timestamp=datetime.now(UTC),
            image=img,
        )
        assert f.width == 640
        assert f.height == 480
        assert f.shape == (480, 640)
