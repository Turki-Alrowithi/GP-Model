"""Tests for ThreadedFrameReader — background capture with prompt shutdown."""

from __future__ import annotations

import time
from collections.abc import Iterator
from datetime import UTC, datetime
from itertools import islice
from threading import Event

import numpy as np

from gpmodel.core.types import Frame
from gpmodel.sources.threaded import ThreadedFrameReader


class _SlowBlockingSource:
    """A source whose `frames()` sleeps on every read, simulating AVFoundation.

    Shutdown must not wait for the sleep to finish — that's the whole
    point of running capture in its own thread.
    """

    def __init__(self, stream_id: str = "slow-cam", delay_s: float = 1.0) -> None:
        self.stream_id = stream_id
        self.delay_s = delay_s
        self._closed = Event()
        self.opened_count = 0
        self.closed_count = 0

    def open(self) -> None:
        self.opened_count += 1

    def close(self) -> None:
        self.closed_count += 1
        self._closed.set()

    @property
    def is_open(self) -> bool:
        return not self._closed.is_set()

    def frames(self) -> Iterator[Frame]:
        i = 0
        img = np.zeros((16, 16, 3), dtype=np.uint8)
        while not self._closed.is_set():
            i += 1
            # Simulate a blocking native read that ignores stop signals.
            if self._closed.wait(timeout=self.delay_s):
                break
            yield Frame(
                stream_id=self.stream_id,
                frame_id=i,
                timestamp=datetime.now(UTC),
                image=img,
            )


class _FastFiniteSource:
    """Produces N frames as fast as possible, then ends."""

    def __init__(self, count: int = 5, stream_id: str = "fast-cam") -> None:
        self.stream_id = stream_id
        self.count = count
        self._opened = False
        self._closed = False

    def open(self) -> None:
        self._opened = True

    def close(self) -> None:
        self._closed = True

    @property
    def is_open(self) -> bool:
        return self._opened and not self._closed

    def frames(self) -> Iterator[Frame]:
        img = np.zeros((16, 16, 3), dtype=np.uint8)
        for i in range(self.count):
            yield Frame(
                stream_id=self.stream_id,
                frame_id=i + 1,
                timestamp=datetime.now(UTC),
                image=img,
            )


def test_finite_source_drains_to_completion() -> None:
    # Queue big enough to hold every frame so the fast source can't
    # backpressure-drop any — this test is about termination, not dropping.
    src = _FastFiniteSource(count=5)
    reader = ThreadedFrameReader(src, queue_size=16)
    reader.start()

    frames = list(islice(reader.frames(), 10))  # allow overflow; we expect 5

    assert len(frames) == 5
    assert [f.frame_id for f in frames] == [1, 2, 3, 4, 5]
    reader.stop()


def test_stop_is_prompt_even_with_blocking_read() -> None:
    """Key test: stop() must return quickly even when the underlying
    source is stuck inside a slow native read."""
    src = _SlowBlockingSource(delay_s=5.0)  # source would block for 5s per frame
    reader = ThreadedFrameReader(src)
    reader.start()

    time.sleep(0.1)  # let the reader thread enter its blocking wait

    t0 = time.perf_counter()
    reader.stop(timeout=2.0)
    elapsed = time.perf_counter() - t0

    assert elapsed < 1.0, f"stop() took {elapsed:.2f}s — should have been near-instant"
    assert src.closed_count >= 1


def test_stop_is_idempotent() -> None:
    src = _FastFiniteSource(count=3)
    reader = ThreadedFrameReader(src)
    reader.start()
    reader.stop()
    reader.stop()  # must not raise


def test_drops_stale_frames_under_backpressure() -> None:
    """Consumer iterates slowly; reader must prefer fresh frames over backlog.

    Uses a paced source (1 ms/frame) against a slow consumer (20 ms/frame)
    so the reader is clearly ahead.
    """

    class _PacedSource:
        stream_id = "paced"

        def __init__(self) -> None:
            self._closed = Event()

        def open(self) -> None: ...
        def close(self) -> None:
            self._closed.set()

        @property
        def is_open(self) -> bool:
            return not self._closed.is_set()

        def frames(self) -> Iterator[Frame]:
            img = np.zeros((8, 8, 3), dtype=np.uint8)
            i = 0
            while not self._closed.is_set():
                i += 1
                if self._closed.wait(timeout=0.001):
                    break
                yield Frame(
                    stream_id=self.stream_id,
                    frame_id=i,
                    timestamp=datetime.now(UTC),
                    image=img,
                )

    src = _PacedSource()
    reader = ThreadedFrameReader(src, queue_size=2)
    reader.start()

    collected: list[int] = []
    for frame in reader.frames():
        collected.append(frame.frame_id)
        time.sleep(0.02)  # 20ms consumer — 20x slower than reader
        if len(collected) >= 5:
            break
    reader.stop()

    # Reader is producing faster than consumer; with maxsize=2 we must see
    # frame ids that skip ahead, not a contiguous 1..5.
    assert collected[-1] > len(collected), f"expected skipping, got {collected}"


def test_stream_id_is_exposed() -> None:
    src = _FastFiniteSource(stream_id="cam-42")
    reader = ThreadedFrameReader(src)
    assert reader.stream_id == "cam-42"
