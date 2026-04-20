"""Tests for GeofenceRule."""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pytest

from gpmodel.core.events import AlertSeverity
from gpmodel.core.types import BBox, Detection, Frame, Track
from gpmodel.rules.geofence import Geofence, GeofenceRule

# Frame is 1280x720; zone covers the right half (normalized).
RIGHT_HALF = Geofence(
    name="right-half",
    points=((0.5, 0.0), (1.0, 0.0), (1.0, 1.0), (0.5, 1.0)),
    normalized=True,
)


def _frame() -> Frame:
    return Frame(
        stream_id="cam-1",
        frame_id=42,
        timestamp=datetime.now(UTC),
        image=np.zeros((720, 1280, 3), dtype=np.uint8),
    )


def _track(track_id: int, x1: float, y1: float, x2: float, y2: float, cls: str = "person") -> Track:
    return Track(
        track_id=track_id,
        class_id=0,
        class_name=cls,
        confidence=0.9,
        bbox=BBox(x1, y1, x2, y2),
        age=10,
        time_since_update=0,
    )


def test_no_tracks_yields_no_alerts(sample_frame: Frame) -> None:
    rule = GeofenceRule(zones=[RIGHT_HALF])
    assert rule.evaluate(sample_frame, [], []) == []


def test_no_zones_yields_no_alerts(sample_frame: Frame) -> None:
    rule = GeofenceRule(zones=[])
    tracks = [_track(1, 800, 300, 900, 500)]
    assert rule.evaluate(sample_frame, [], tracks) == []


def test_track_outside_any_zone_is_ignored(sample_frame: Frame) -> None:
    rule = GeofenceRule(zones=[RIGHT_HALF])
    # Track entirely in the left half — feet at x=300, which is < 640
    tracks = [_track(1, 200, 200, 400, 600)]
    assert rule.evaluate(sample_frame, [], tracks) == []


def test_track_inside_zone_fires_alert(sample_frame: Frame) -> None:
    rule = GeofenceRule(zones=[RIGHT_HALF])
    tracks = [_track(1, 800, 300, 900, 500)]  # foot at (850, 500)
    alerts = rule.evaluate(sample_frame, [], tracks)
    assert len(alerts) == 1
    assert alerts[0].rule_type == "geofence_breach"
    assert alerts[0].severity == AlertSeverity.HIGH
    assert alerts[0].evidence["zone"] == "right-half"
    assert alerts[0].evidence["track_id"] == 1


def test_classes_filter_excludes_non_matching(sample_frame: Frame) -> None:
    rule = GeofenceRule(zones=[RIGHT_HALF], classes=frozenset({"person"}))
    tracks = [_track(1, 800, 300, 900, 500, cls="car")]
    assert rule.evaluate(sample_frame, [], tracks) == []


def test_cooldown_suppresses_repeated_alerts_for_same_track(sample_frame: Frame) -> None:
    rule = GeofenceRule(zones=[RIGHT_HALF], cooldown_s=60.0)
    tracks = [_track(1, 800, 300, 900, 500)]
    first = rule.evaluate(sample_frame, [], tracks)
    second = rule.evaluate(sample_frame, [], tracks)
    third = rule.evaluate(sample_frame, [], tracks)

    assert len(first) == 1
    assert second == []
    assert third == []


def test_different_tracks_alert_independently(sample_frame: Frame) -> None:
    rule = GeofenceRule(zones=[RIGHT_HALF], cooldown_s=60.0)
    tracks = [_track(1, 800, 300, 900, 500), _track(2, 1000, 100, 1100, 300)]
    alerts = rule.evaluate(sample_frame, [], tracks)
    ids = sorted(a.evidence["track_id"] for a in alerts)
    assert ids == [1, 2]


def test_foot_point_vs_center_differ_at_horizontal_boundary() -> None:
    """foot_point only matters when the fence is horizontal-ish — same x
    for feet and center, different y."""
    frame = _frame()  # 1280x720
    bottom_half = Geofence(
        name="bottom",
        points=((0.0, 0.5), (1.0, 0.5), (1.0, 1.0), (0.0, 1.0)),
        normalized=True,
    )
    # bbox: center y = 300 (above line at 360), feet y = 400 (below line)
    tracks = [_track(1, 600, 200, 700, 400)]

    fires = GeofenceRule(zones=[bottom_half], foot_point=True).evaluate(frame, [], tracks)
    silent = GeofenceRule(zones=[bottom_half], foot_point=False).evaluate(frame, [], tracks)

    assert len(fires) == 1
    assert silent == []


def test_pixel_coordinates_mode() -> None:
    # Explicit pixel-coordinate zone.
    zone = Geofence(
        name="pixels",
        points=((600, 0), (1280, 0), (1280, 720), (600, 720)),
        normalized=False,
    )
    frame = _frame()
    tracks_inside = [_track(1, 800, 300, 900, 500)]
    tracks_outside = [_track(2, 100, 300, 200, 500)]

    rule = GeofenceRule(zones=[zone])
    assert len(rule.evaluate(frame, [], tracks_inside)) == 1
    # Reset cooldown implicitly: different track_id.
    assert rule.evaluate(frame, [], tracks_outside) == []


def test_severity_is_configurable(sample_frame: Frame) -> None:
    rule = GeofenceRule(zones=[RIGHT_HALF], severity=AlertSeverity.CRITICAL)
    tracks = [_track(1, 800, 300, 900, 500)]
    alerts = rule.evaluate(sample_frame, [], tracks)
    assert alerts[0].severity == AlertSeverity.CRITICAL


# Silence ruff's unused-import warnings when running only this file.
_ = pytest
_ = Detection
