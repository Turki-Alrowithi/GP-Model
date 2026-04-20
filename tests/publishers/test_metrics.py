"""Tests for MetricsSubscriber."""

from __future__ import annotations

from gpmodel.core.events import AlertRaised, AlertSeverity, PerfSampled
from gpmodel.core.types import PerfSample
from gpmodel.publishers.metrics import MetricsSubscriber


def test_empty_summary() -> None:
    m = MetricsSubscriber()
    s = m.summary()
    assert s.frames == 0
    assert s.avg_fps == 0.0
    assert s.alerts_total == 0


def test_averages_perf_samples_and_remembers_frame_count() -> None:
    m = MetricsSubscriber()
    for fps, lat, frames in [(30.0, 33.0, 30), (25.0, 40.0, 60)]:
        m.on_event(
            PerfSampled(
                stream_id="cam-1",
                sample=PerfSample(
                    stream_id="cam-1",
                    fps=fps,
                    latency_ms=lat,
                    frame_count=frames,
                    dropped_frames=0,
                ),
            )
        )
    s = m.summary()
    assert s.perf_samples == 2
    assert s.avg_fps == 27.5
    assert s.avg_latency_ms == 36.5
    assert s.frames == 60


def test_counts_alerts_by_rule_and_severity() -> None:
    m = MetricsSubscriber()
    m.on_event(AlertRaised(stream_id="cam-1", severity=AlertSeverity.HIGH, rule_type="intruder"))
    m.on_event(AlertRaised(stream_id="cam-1", severity=AlertSeverity.HIGH, rule_type="intruder"))
    m.on_event(AlertRaised(stream_id="cam-1", severity=AlertSeverity.MEDIUM, rule_type="crowd"))

    s = m.summary()
    assert s.alerts_total == 3
    assert s.alerts_by_rule == {"intruder": 2, "crowd": 1}
    assert s.alerts_by_severity == {"HIGH": 2, "MEDIUM": 1}
