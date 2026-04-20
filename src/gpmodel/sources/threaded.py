"""Threaded frame reader — decouples frame capture from the detection loop.

Some video sources (macOS AVFoundation, RTSP) block in native code during
`read()` and swallow Python signals until the next frame arrives. When
that happens the main loop can't see Ctrl-C for seconds at a time.

By moving the read into a worker thread and communicating through a
bounded queue, the main loop polls with a short timeout and can honour
a stop request immediately — regardless of what the underlying source
is doing.

As a bonus, the bounded queue drops stale frames under load (preferring
freshness over backlog), which is exactly what a real-time surveillance
system wants.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import suppress
from queue import Empty, Full, Queue
from threading import Event, Thread

from gpmodel.core.interfaces import VideoSource
from gpmodel.core.types import Frame

logger = logging.getLogger(__name__)


class ThreadedFrameReader:
    """Background reader that pumps frames from a VideoSource into a queue.

    Usage:
        reader = ThreadedFrameReader(source)
        reader.start()
        for frame in reader.frames():
            ...   # main loop
        reader.stop()   # idempotent, releases the source
    """

    def __init__(
        self,
        source: VideoSource,
        queue_size: int = 2,
        poll_timeout: float = 0.1,
    ) -> None:
        self._source = source
        self._queue: Queue[Frame | None] = Queue(maxsize=queue_size)
        self._stop = Event()
        self._thread: Thread | None = None
        self._poll_timeout = poll_timeout

    # ── Lifecycle ──────────────────────────────────────────
    def start(self) -> None:
        if self._thread is not None:
            return
        self._source.open()
        self._thread = Thread(
            target=self._pump,
            name=f"frame-reader-{self._source.stream_id}",
            daemon=True,
        )
        self._thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        """Release the source and join the worker. Safe to call multiple times."""
        if self._stop.is_set():
            return
        self._stop.set()
        # Closing the source unblocks any pending cv2.VideoCapture.read() by
        # releasing the underlying device handle — the worker will then see
        # EOF and exit promptly.
        try:
            self._source.close()
        except Exception:
            logger.exception("Source close failed during stop()")
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None

    @property
    def stream_id(self) -> str:
        return self._source.stream_id

    # ── Iteration ──────────────────────────────────────────
    def frames(self) -> Iterator[Frame]:
        if self._thread is None:
            self.start()

        while not self._stop.is_set():
            try:
                frame = self._queue.get(timeout=self._poll_timeout)
            except Empty:
                continue
            if frame is None:  # reader thread signalled EOF / error
                break
            yield frame

    # ── Internals ──────────────────────────────────────────
    def _pump(self) -> None:
        try:
            for frame in self._source.frames():
                if self._stop.is_set():
                    break
                # Prefer freshness: if the consumer is lagging, drop the oldest
                # frame rather than building a backlog. We only put with zero
                # timeout so the reader is never slowed down to consumer speed.
                try:
                    self._queue.put_nowait(frame)
                except Full:
                    with suppress(Empty):
                        self._queue.get_nowait()
                    with suppress(Full):
                        self._queue.put_nowait(frame)
        except Exception:
            logger.exception("Reader thread crashed for '%s'", self._source.stream_id)
        finally:
            # Sentinel so the consumer's frames() loop exits promptly.
            # The queue may be full — drop one item to make room.
            while True:
                try:
                    self._queue.put_nowait(None)
                    break
                except Full:
                    try:
                        self._queue.get_nowait()
                    except Empty:
                        break  # pathological; prevent infinite loop
