"""Tests for PerfMeter."""

from __future__ import annotations

from gpmodel.telemetry.perf import PerfMeter


def test_snapshot_zero_when_empty() -> None:
    m = PerfMeter("cam-1")
    snap = m.snapshot()
    assert snap.fps == 0.0
    assert snap.latency_ms == 0.0
    assert snap.frame_count == 0


def test_fps_and_latency_average() -> None:
    m = PerfMeter("cam-1", window=10)
    for lat in [40.0, 50.0, 60.0]:  # avg 50 ms → 20 FPS
        m.tick(lat)

    snap = m.snapshot()
    assert snap.frame_count == 3
    assert snap.latency_ms == 50.0
    assert snap.fps == 20.0


def test_rolling_window_drops_old_samples() -> None:
    m = PerfMeter("cam-1", window=3)
    for lat in [1000.0, 1000.0, 1000.0, 10.0]:  # only last 3 retained
        m.tick(lat)

    snap = m.snapshot()
    # avg of (1000, 1000, 10) = 670 ms → ~1.49 FPS
    assert snap.latency_ms == (1000 + 1000 + 10) / 3
    assert snap.fps == 1000.0 / snap.latency_ms


def test_should_emit_every_n_ticks() -> None:
    m = PerfMeter("cam-1", emit_every=5)
    emits = [m.should_emit() for _ in range(11) if (m.tick(10.0) or True)]
    # pattern: False False False False True (after 5), False x4, True (after 10), False
    assert emits == [False, False, False, False, True, False, False, False, False, True, False]


def test_mark_dropped() -> None:
    m = PerfMeter("cam-1")
    m.mark_dropped(2)
    m.tick(10.0)
    assert m.snapshot().dropped_frames == 2
