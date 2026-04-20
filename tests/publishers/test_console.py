"""Tests for ConsoleSubscriber — assert against captured rich output."""

from __future__ import annotations

import io

from rich.console import Console

from gpmodel.core.events import (
    AlertRaised,
    AlertSeverity,
    DetectionsReady,
    PerfSampled,
    StreamStateChanged,
)
from gpmodel.core.types import PerfSample
from gpmodel.publishers.console import ConsoleSubscriber


def _capture() -> tuple[ConsoleSubscriber, io.StringIO]:
    buf = io.StringIO()
    console = Console(file=buf, force_terminal=False, width=200)
    return ConsoleSubscriber(console=console), buf


def test_prints_alert_with_severity() -> None:
    sub, buf = _capture()
    sub.on_event(
        AlertRaised(
            stream_id="cam-1",
            severity=AlertSeverity.CRITICAL,
            rule_type="weapon",
            title="weapon detected",
        )
    )
    out = buf.getvalue()
    assert "CRITICAL" in out
    assert "weapon" in out
    assert "cam-1" in out


def test_prints_perf_snapshot() -> None:
    sub, buf = _capture()
    sub.on_event(
        PerfSampled(
            stream_id="cam-1",
            sample=PerfSample(
                stream_id="cam-1",
                fps=30.0,
                latency_ms=33.3,
                frame_count=120,
                dropped_frames=0,
            ),
        )
    )
    out = buf.getvalue()
    assert "PERF" in out
    assert "30.0" in out
    assert "33.3" in out


def test_prints_state_change() -> None:
    sub, buf = _capture()
    sub.on_event(StreamStateChanged(stream_id="cam-1", state="opened"))
    sub.on_event(StreamStateChanged(stream_id="cam-1", state="error", detail="disconnected"))

    out = buf.getvalue()
    assert "opened" in out
    assert "error" in out
    assert "disconnected" in out


def test_detections_silent_by_default() -> None:
    sub, buf = _capture()
    sub.on_event(DetectionsReady(stream_id="cam-1"))
    assert buf.getvalue() == ""


def test_detections_print_when_enabled() -> None:
    buf = io.StringIO()
    console = Console(file=buf, force_terminal=False, width=200)
    sub = ConsoleSubscriber(print_detections=True, console=console)
    from gpmodel.core.types import BBox, Detection

    sub.on_event(
        DetectionsReady(
            stream_id="cam-1",
            detections=(
                Detection(class_id=0, class_name="person", confidence=0.9, bbox=BBox(0, 0, 1, 1)),
                Detection(class_id=0, class_name="person", confidence=0.8, bbox=BBox(0, 0, 2, 2)),
            ),
        )
    )
    out = buf.getvalue()
    assert "detections=2" in out
    assert "person=2" in out
    assert "frame#" in out


def test_detections_with_tracks_show_track_ids() -> None:
    """When tracks are present, the summary lists track ids per class.

    This is the operator-facing cue that ByteTrack considers successive
    frames as the *same* object — no alert dedup anxiety.
    """
    buf = io.StringIO()
    console = Console(file=buf, force_terminal=False, width=200)
    sub = ConsoleSubscriber(print_detections=True, console=console)
    from gpmodel.core.types import BBox, Detection, Track

    sub.on_event(
        DetectionsReady(
            stream_id="cam-1",
            detections=(
                Detection(class_id=0, class_name="person", confidence=0.9, bbox=BBox(0, 0, 1, 1)),
                Detection(class_id=0, class_name="person", confidence=0.8, bbox=BBox(0, 0, 2, 2)),
                Detection(class_id=2, class_name="car", confidence=0.7, bbox=BBox(0, 0, 3, 3)),
            ),
            tracks=(
                Track(
                    track_id=3,
                    class_id=0,
                    class_name="person",
                    confidence=0.9,
                    bbox=BBox(0, 0, 1, 1),
                    age=120,
                    time_since_update=0,
                ),
                Track(
                    track_id=7,
                    class_id=0,
                    class_name="person",
                    confidence=0.8,
                    bbox=BBox(0, 0, 2, 2),
                    age=45,
                    time_since_update=0,
                ),
                Track(
                    track_id=11,
                    class_id=2,
                    class_name="car",
                    confidence=0.7,
                    bbox=BBox(0, 0, 3, 3),
                    age=20,
                    time_since_update=0,
                ),
            ),
        )
    )
    out = buf.getvalue()
    assert "person#3,7" in out
    assert "car#11" in out
    # raw counts should not appear when tracks take over
    assert "person=2" not in out
