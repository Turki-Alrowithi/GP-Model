"""Tests for InferenceEngine — driven by fake source/detector/tracker."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime

import numpy as np
import pytest

from gpmodel.core.dispatcher import AlertDispatcher
from gpmodel.core.events import DetectionsReady, Event, PerfSampled, StreamStateChanged
from gpmodel.core.types import BBox, Detection, Frame, Track
from gpmodel.pipeline.engine import InferenceEngine


# ── Fake components ────────────────────────────────────────
@dataclass
class FakeSource:
    stream_id: str = "cam-1"
    count: int = 5
    opened: bool = False
    closed: bool = False

    def open(self) -> None:
        self.opened = True

    def close(self) -> None:
        self.closed = True

    @property
    def is_open(self) -> bool:
        return self.opened and not self.closed

    def frames(self) -> Iterator[Frame]:
        img = np.zeros((64, 64, 3), dtype=np.uint8)
        for i in range(self.count):
            yield Frame(
                stream_id=self.stream_id,
                frame_id=i + 1,
                timestamp=datetime.now(UTC),
                image=img,
            )


@dataclass
class FakeDetector:
    warmed: bool = False
    calls: int = 0

    def warmup(self) -> None:
        self.warmed = True

    def detect(self, frame: Frame) -> list[Detection]:
        self.calls += 1
        return [
            Detection(class_id=0, class_name="person", confidence=0.9, bbox=BBox(0, 0, 10, 10))
        ]

    def close(self) -> None: ...


@dataclass
class FakeTracker:
    calls: int = 0

    def update(self, detections: list[Detection], frame: Frame) -> list[Track]:
        self.calls += 1
        return [
            Track(
                track_id=i + 1,
                class_id=d.class_id,
                class_name=d.class_name,
                confidence=d.confidence,
                bbox=d.bbox,
                age=1,
                time_since_update=0,
            )
            for i, d in enumerate(detections)
        ]

    def reset(self) -> None: ...


@dataclass
class Recorder:
    events: list[Event] = field(default_factory=list)

    def on_event(self, event: Event) -> None:
        self.events.append(event)


# ── Tests ──────────────────────────────────────────────────
def test_warmup_runs_before_source_opens() -> None:
    src = FakeSource(count=1)
    det = FakeDetector()
    bus = AlertDispatcher()
    eng = InferenceEngine("cam-1", src, det, bus, threaded_reader=False)

    eng.run()
    assert det.warmed


def test_emits_detectionsready_per_frame() -> None:
    src = FakeSource(count=3)
    det = FakeDetector()
    bus = AlertDispatcher()
    rec = Recorder()
    bus.subscribe(rec)

    InferenceEngine("cam-1", src, det, bus, tracker=FakeTracker(), threaded_reader=False).run()

    det_events = [e for e in rec.events if isinstance(e, DetectionsReady)]
    assert len(det_events) == 3
    assert all(len(e.detections) == 1 for e in det_events)
    assert all(len(e.tracks) == 1 for e in det_events)


def test_emits_stream_state_open_and_closed() -> None:
    src = FakeSource(count=2)
    det = FakeDetector()
    bus = AlertDispatcher()
    rec = Recorder()
    bus.subscribe(rec)

    InferenceEngine("cam-1", src, det, bus, threaded_reader=False).run()

    state_events = [e for e in rec.events if isinstance(e, StreamStateChanged)]
    states = [e.state for e in state_events]
    assert "opened" in states
    assert "closed" in states


def test_emits_perf_sampled_on_interval() -> None:
    src = FakeSource(count=10)
    det = FakeDetector()
    bus = AlertDispatcher()
    rec = Recorder()
    bus.subscribe(rec)

    InferenceEngine("cam-1", src, det, bus, perf_emit_every=3, threaded_reader=False).run()

    perf_events = [e for e in rec.events if isinstance(e, PerfSampled)]
    # 10 frames, emit every 3 → 3 perf events (at frame 3, 6, 9)
    assert len(perf_events) == 3
    assert all(p.sample is not None for p in perf_events)


def test_stop_aborts_loop() -> None:
    src = FakeSource(count=100)
    det = FakeDetector()
    bus = AlertDispatcher()

    class StoppingSubscriber:
        def __init__(self) -> None:
            self.seen = 0

        def on_event(self, event: Event) -> None:
            if isinstance(event, DetectionsReady):
                self.seen += 1
                if self.seen == 2:
                    eng.stop()

    eng = InferenceEngine("cam-1", src, det, bus, threaded_reader=False)
    bus.subscribe(StoppingSubscriber())
    eng.run()

    assert det.calls <= 3  # 2 full iterations + possibly 1 more before stop check


def test_closes_resources_on_error() -> None:
    src = FakeSource(count=5)

    class BrokenDetector(FakeDetector):
        def detect(self, frame: Frame) -> list[Detection]:
            raise RuntimeError("boom")

    det = BrokenDetector()
    bus = AlertDispatcher()
    rec = Recorder()
    bus.subscribe(rec)
    eng = InferenceEngine("cam-1", src, det, bus, threaded_reader=False)

    with pytest.raises(RuntimeError, match="boom"):
        eng.run()

    assert src.closed
    states = [e.state for e in rec.events if isinstance(e, StreamStateChanged)]
    assert "error" in states
    assert "closed" in states
