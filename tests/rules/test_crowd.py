"""Tests for CrowdRule."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np

from gpmodel.core.events import AlertSeverity
from gpmodel.core.types import BBox, Frame, Track
from gpmodel.rules.crowd import CrowdRule
from gpmodel.rules.geofence import Geofence


def _frame(t: datetime, frame_id: int = 1) -> Frame:
    return Frame(
        stream_id="cam-1",
        frame_id=frame_id,
        timestamp=t,
        image=np.zeros((720, 1280, 3), dtype=np.uint8),
    )


def _track(track_id: int, cls: str = "person") -> Track:
    # arbitrary box near frame center
    return Track(
        track_id=track_id,
        class_id=0,
        class_name=cls,
        confidence=0.9,
        bbox=BBox(600 + track_id, 300, 700 + track_id, 500),
        age=10,
        time_since_update=0,
    )


def test_below_threshold_never_fires() -> None:
    rule = CrowdRule(threshold=3, min_duration_s=0.0)
    t0 = datetime(2025, 1, 1, tzinfo=UTC)
    alerts = rule.evaluate(_frame(t0), [], [_track(1), _track(2)])
    assert alerts == []


def test_threshold_with_zero_duration_fires_immediately() -> None:
    rule = CrowdRule(threshold=3, min_duration_s=0.0)
    t0 = datetime(2025, 1, 1, tzinfo=UTC)

    # First call arms the timer; duration 0 means next call (or even same)
    # crosses the threshold. The rule uses t_now - t_since_start, so a second
    # call at the same instant yields elapsed=0 >= min_duration=0.
    rule.evaluate(_frame(t0, 1), [], [_track(1), _track(2), _track(3)])
    alerts = rule.evaluate(_frame(t0, 2), [], [_track(1), _track(2), _track(3)])

    assert len(alerts) == 1
    assert alerts[0].rule_type == "crowd_formed"
    assert alerts[0].evidence["count"] == 3
    assert alerts[0].severity == AlertSeverity.MEDIUM


def test_requires_sustained_duration() -> None:
    rule = CrowdRule(threshold=3, min_duration_s=2.0)
    t0 = datetime(2025, 1, 1, tzinfo=UTC)
    tracks = [_track(1), _track(2), _track(3)]

    # Only 1s elapsed — no alert yet.
    assert rule.evaluate(_frame(t0, 1), [], tracks) == []
    assert rule.evaluate(_frame(t0 + timedelta(seconds=1.0), 2), [], tracks) == []
    alerts = rule.evaluate(_frame(t0 + timedelta(seconds=2.5), 3), [], tracks)
    assert len(alerts) == 1


def test_timer_resets_when_count_drops() -> None:
    rule = CrowdRule(threshold=3, min_duration_s=2.0)
    t0 = datetime(2025, 1, 1, tzinfo=UTC)
    many = [_track(1), _track(2), _track(3)]
    few = [_track(1)]

    rule.evaluate(_frame(t0, 1), [], many)
    # Crowd breaks up — timer should reset.
    rule.evaluate(_frame(t0 + timedelta(seconds=1.0), 2), [], few)
    # Crowd reforms; first frame re-arms the timer, not fire.
    assert rule.evaluate(_frame(t0 + timedelta(seconds=1.5), 3), [], many) == []
    # 0.5s after re-arming — still below 2s threshold.
    assert rule.evaluate(_frame(t0 + timedelta(seconds=2.0), 4), [], many) == []
    # 3s after re-arming — fires.
    assert len(rule.evaluate(_frame(t0 + timedelta(seconds=4.5), 5), [], many)) == 1


def test_classes_filter() -> None:
    rule = CrowdRule(threshold=2, min_duration_s=0.0, classes=frozenset({"person"}))
    t0 = datetime(2025, 1, 1, tzinfo=UTC)
    tracks = [_track(1, cls="car"), _track(2, cls="car"), _track(3, cls="person")]

    # Only 1 person — under threshold.
    assert rule.evaluate(_frame(t0, 1), [], tracks) == []
    assert rule.evaluate(_frame(t0, 2), [], tracks) == []


def test_zone_filter() -> None:
    # Right-half zone; crowd tracks are near x=600-700 (mostly left of fence)
    right_half = Geofence(
        name="right",
        points=((0.5, 0.0), (1.0, 0.0), (1.0, 1.0), (0.5, 1.0)),
    )
    rule = CrowdRule(threshold=2, min_duration_s=0.0, zone=right_half)
    t0 = datetime(2025, 1, 1, tzinfo=UTC)
    # bbox centers: x=650, 651, 652 — feet at same x. 1280*0.5=640, so feet>640 is inside.
    tracks = [_track(i) for i in (1, 2, 3)]
    rule.evaluate(_frame(t0, 1), [], tracks)
    alerts = rule.evaluate(_frame(t0, 2), [], tracks)
    assert len(alerts) == 1
    assert alerts[0].evidence["zone"] == "right"


def test_cooldown_prevents_immediate_refire() -> None:
    rule = CrowdRule(threshold=2, min_duration_s=0.0, cooldown_s=30.0)
    t0 = datetime(2025, 1, 1, tzinfo=UTC)
    tracks = [_track(1), _track(2)]

    rule.evaluate(_frame(t0, 1), [], tracks)
    first = rule.evaluate(_frame(t0, 2), [], tracks)
    second = rule.evaluate(_frame(t0 + timedelta(seconds=5), 3), [], tracks)
    assert len(first) == 1
    assert second == []
