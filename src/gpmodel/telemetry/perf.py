"""Rolling performance meter — FPS and per-frame latency.

Tracks a bounded window of recent frame timings so the reported
numbers reflect current conditions, not a lifetime average.
"""

from __future__ import annotations

from collections import deque

from gpmodel.core.types import PerfSample


class PerfMeter:
    """Rolling FPS / latency meter for a single stream.

    `tick()` records one processed frame; `snapshot()` returns the
    current state without resetting.
    """

    def __init__(
        self,
        stream_id: str,
        window: int = 60,
        emit_every: int = 30,
    ) -> None:
        self.stream_id = stream_id
        self._window = window
        self._emit_every = emit_every
        self._latencies_ms: deque[float] = deque(maxlen=window)
        self._frame_count: int = 0
        self._dropped: int = 0
        self._frames_since_emit: int = 0

    def tick(self, latency_ms: float) -> None:
        self._latencies_ms.append(latency_ms)
        self._frame_count += 1
        self._frames_since_emit += 1

    def mark_dropped(self, n: int = 1) -> None:
        self._dropped += n

    def should_emit(self) -> bool:
        if self._frames_since_emit < self._emit_every:
            return False
        self._frames_since_emit = 0
        return True

    def snapshot(self) -> PerfSample:
        if self._latencies_ms:
            avg_latency = sum(self._latencies_ms) / len(self._latencies_ms)
            fps = 1000.0 / avg_latency if avg_latency > 0 else 0.0
        else:
            avg_latency = 0.0
            fps = 0.0
        return PerfSample(
            stream_id=self.stream_id,
            fps=fps,
            latency_ms=avg_latency,
            frame_count=self._frame_count,
            dropped_frames=self._dropped,
        )
