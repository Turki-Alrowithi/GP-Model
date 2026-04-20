"""Tests for the AlertDispatcher (Observer hub)."""

from __future__ import annotations

from dataclasses import dataclass, field

from gpmodel.core.dispatcher import AlertDispatcher
from gpmodel.core.events import AlertRaised, AlertSeverity, Event, PerfSampled


@dataclass
class RecordingSubscriber:
    received: list[Event] = field(default_factory=list)

    def on_event(self, event: Event) -> None:
        self.received.append(event)


class BrokenSubscriber:
    def on_event(self, event: Event) -> None:
        raise RuntimeError("boom")


def test_single_subscriber_receives_event() -> None:
    bus = AlertDispatcher()
    sub = RecordingSubscriber()
    bus.subscribe(sub)

    event = AlertRaised(
        stream_id="cam-1",
        severity=AlertSeverity.HIGH,
        rule_type="intruder",
        title="unauthorized person",
    )
    bus.publish(event)

    assert sub.received == [event]


def test_multiple_subscribers_all_notified() -> None:
    bus = AlertDispatcher()
    a = RecordingSubscriber()
    b = RecordingSubscriber()
    bus.subscribe(a)
    bus.subscribe(b)

    event = PerfSampled(stream_id="cam-1")
    bus.publish(event)

    assert a.received == [event]
    assert b.received == [event]


def test_unsubscribe_stops_delivery() -> None:
    bus = AlertDispatcher()
    sub = RecordingSubscriber()
    bus.subscribe(sub)
    bus.unsubscribe(sub)

    bus.publish(PerfSampled(stream_id="cam-1"))

    assert sub.received == []


def test_double_subscribe_is_idempotent() -> None:
    bus = AlertDispatcher()
    sub = RecordingSubscriber()
    bus.subscribe(sub)
    bus.subscribe(sub)
    assert bus.subscriber_count == 1


def test_exception_in_one_subscriber_does_not_block_others() -> None:
    bus = AlertDispatcher()
    broken = BrokenSubscriber()
    good = RecordingSubscriber()
    bus.subscribe(broken)
    bus.subscribe(good)

    event = PerfSampled(stream_id="cam-1")
    bus.publish(event)

    assert good.received == [event]
