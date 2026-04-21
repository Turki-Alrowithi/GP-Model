"""Tests for WeaponRule."""

from __future__ import annotations

from gpmodel.core.events import AlertSeverity
from gpmodel.core.types import BBox, Frame, Track
from gpmodel.rules.weapon import WeaponRule


def _track(
    tid: int,
    cls: str = "Knife",
    conf: float = 0.9,
    age: int = 5,
) -> Track:
    return Track(
        track_id=tid,
        class_id=0,
        class_name=cls,
        confidence=conf,
        bbox=BBox(0, 0, 10, 10),
        age=age,
        time_since_update=0,
    )


def test_fires_on_sustained_weapon(sample_frame: Frame) -> None:
    rule = WeaponRule(classes=frozenset({"Knife"}), min_consecutive_frames=3, min_confidence=0.5)
    alerts = rule.evaluate(sample_frame, [], [_track(1)])

    assert len(alerts) == 1
    assert alerts[0].rule_type == "weapon_detected"
    assert alerts[0].severity == AlertSeverity.CRITICAL
    assert alerts[0].evidence["class_name"] == "Knife"
    assert alerts[0].evidence["track_id"] == 1


def test_ignores_non_weapon_classes(sample_frame: Frame) -> None:
    rule = WeaponRule(classes=frozenset({"Knife"}))
    assert rule.evaluate(sample_frame, [], [_track(1, cls="person")]) == []


def test_below_confidence_threshold_is_ignored(sample_frame: Frame) -> None:
    rule = WeaponRule(min_confidence=0.8)
    assert rule.evaluate(sample_frame, [], [_track(1, conf=0.7)]) == []


def test_new_track_below_min_consecutive_frames(sample_frame: Frame) -> None:
    rule = WeaponRule(min_consecutive_frames=5)
    assert rule.evaluate(sample_frame, [], [_track(1, age=2)]) == []


def test_cooldown_suppresses_repeat_for_same_track(sample_frame: Frame) -> None:
    rule = WeaponRule(cooldown_s=60.0)
    first = rule.evaluate(sample_frame, [], [_track(1)])
    second = rule.evaluate(sample_frame, [], [_track(1)])

    assert len(first) == 1
    assert second == []


def test_different_tracks_alert_independently(sample_frame: Frame) -> None:
    rule = WeaponRule(cooldown_s=60.0)
    alerts = rule.evaluate(sample_frame, [], [_track(1), _track(2)])
    assert len(alerts) == 2
    ids = sorted(a.evidence["track_id"] for a in alerts)
    assert ids == [1, 2]


def test_supports_multiple_weapon_classes(sample_frame: Frame) -> None:
    rule = WeaponRule(classes=frozenset({"Knife", "Pistol"}))
    alerts = rule.evaluate(
        sample_frame,
        [],
        [_track(1, cls="Knife"), _track(2, cls="Pistol"), _track(3, cls="person")],
    )
    assert len(alerts) == 2
