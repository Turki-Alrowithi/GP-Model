"""Tests for JSONLFileSubscriber."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from gpmodel.core.events import (
    AlertRaised,
    AlertSeverity,
    DetectionsReady,
    PerfSampled,
    StreamStateChanged,
)
from gpmodel.core.types import BBox, Detection, Frame, PerfSample
from gpmodel.publishers.jsonl import JSONLFileSubscriber


def test_writes_alert_as_single_line(tmp_path: Path) -> None:
    log = tmp_path / "events.jsonl"
    with JSONLFileSubscriber(log) as sub:
        sub.on_event(
            AlertRaised(
                stream_id="cam-1",
                severity=AlertSeverity.HIGH,
                rule_type="intruder",
                title="unauthorized",
                description="someone near fence",
                evidence={"zone": "north"},
            )
        )

    lines = log.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["type"] == "AlertRaised"
    assert record["event"]["severity"] == "HIGH"
    assert record["event"]["rule_type"] == "intruder"
    assert record["event"]["evidence"] == {"zone": "north"}


def test_appends_across_calls(tmp_path: Path) -> None:
    log = tmp_path / "events.jsonl"
    with JSONLFileSubscriber(log) as sub:
        sub.on_event(StreamStateChanged(stream_id="cam-1", state="opened"))
        sub.on_event(
            PerfSampled(
                stream_id="cam-1",
                sample=PerfSample(
                    stream_id="cam-1",
                    fps=30.0,
                    latency_ms=33.0,
                    frame_count=30,
                    dropped_frames=0,
                ),
            )
        )

    types = [json.loads(ln)["type"] for ln in log.read_text().splitlines()]
    assert types == ["StreamStateChanged", "PerfSampled"]


def test_omits_detections_by_default(tmp_path: Path) -> None:
    log = tmp_path / "events.jsonl"
    frame = Frame(
        stream_id="cam-1",
        frame_id=1,
        timestamp=datetime.now(UTC),
        image=np.zeros((4, 4, 3), dtype=np.uint8),
    )
    with JSONLFileSubscriber(log) as sub:
        sub.on_event(
            DetectionsReady(
                stream_id="cam-1",
                frame=frame,
                detections=(
                    Detection(
                        class_id=0, class_name="person", confidence=0.9, bbox=BBox(0, 0, 1, 1)
                    ),
                ),
            )
        )

    assert log.read_text() == ""


def test_includes_detections_when_opted_in(tmp_path: Path) -> None:
    log = tmp_path / "events.jsonl"
    frame = Frame(
        stream_id="cam-1",
        frame_id=1,
        timestamp=datetime.now(UTC),
        image=np.zeros((4, 4, 3), dtype=np.uint8),
    )
    with JSONLFileSubscriber(log, include_detection_frames=True) as sub:
        sub.on_event(
            DetectionsReady(
                stream_id="cam-1",
                frame=frame,
                detections=(
                    Detection(
                        class_id=0, class_name="person", confidence=0.9, bbox=BBox(0, 0, 10, 10)
                    ),
                ),
            )
        )

    record = json.loads(log.read_text().splitlines()[0])
    # Image pixels must not be in the record — only a shape descriptor.
    img = record["event"]["frame"]["image"]
    assert "__ndarray__" in img
    assert img["__ndarray__"]["shape"] == [4, 4, 3]


def test_creates_parent_directory(tmp_path: Path) -> None:
    log = tmp_path / "nested" / "dir" / "events.jsonl"
    with JSONLFileSubscriber(log) as sub:
        sub.on_event(StreamStateChanged(stream_id="cam-1", state="closed"))
    assert log.exists()
