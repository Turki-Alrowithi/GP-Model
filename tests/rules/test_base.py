"""Tests for Cooldown and RulesEngine."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from gpmodel.core.events import AlertRaised, AlertSeverity
from gpmodel.core.types import Detection, Frame, Track
from gpmodel.rules.base import Cooldown, Rule, RulesEngine


class _AlwaysFires(Rule):
    name = "always"

    def __init__(self, alert: AlertRaised) -> None:
        self._alert = alert

    def evaluate(
        self, frame: Frame, detections: list[Detection], tracks: list[Track]
    ) -> list[AlertRaised]:
        return [self._alert]


class _NeverFires(Rule):
    name = "never"

    def evaluate(
        self, frame: Frame, detections: list[Detection], tracks: list[Track]
    ) -> list[AlertRaised]:
        return []


def test_cooldown_blocks_within_ttl() -> None:
    cd = Cooldown(seconds=10.0)
    t0 = datetime(2025, 1, 1, tzinfo=UTC)
    assert cd.allow(("k",), now=t0)
    assert not cd.allow(("k",), now=t0 + timedelta(seconds=5))
    assert cd.allow(("k",), now=t0 + timedelta(seconds=11))


def test_cooldown_keys_are_independent() -> None:
    cd = Cooldown(seconds=10.0)
    t0 = datetime(2025, 1, 1, tzinfo=UTC)
    assert cd.allow(("a",), now=t0)
    assert cd.allow(("b",), now=t0)  # different key, not blocked


def test_cooldown_reset_clears_state() -> None:
    cd = Cooldown(seconds=10.0)
    t0 = datetime(2025, 1, 1, tzinfo=UTC)
    cd.allow(("k",), now=t0)
    cd.reset()
    assert cd.allow(("k",), now=t0)


def test_rules_engine_fans_out_to_every_rule(sample_frame: Frame) -> None:
    alert = AlertRaised(
        stream_id="cam-1",
        severity=AlertSeverity.LOW,
        rule_type="always",
    )
    engine = RulesEngine([_AlwaysFires(alert), _NeverFires(), _AlwaysFires(alert)])

    out = engine.evaluate(sample_frame, [], [])
    assert len(out) == 2


def test_rules_engine_add_registers_rule(sample_frame: Frame) -> None:
    engine = RulesEngine()
    engine.add(_AlwaysFires(AlertRaised(stream_id="cam-1")))
    assert len(engine.rules()) == 1
    assert len(engine.evaluate(sample_frame, [], [])) == 1
