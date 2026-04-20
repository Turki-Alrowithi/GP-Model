"""Inference engine — the main orchestrator (Chain of Responsibility).

Wires a VideoSource → Detector → Tracker and publishes results through
an AlertDispatcher. The engine itself has no opinion on what subscribers
do with the events; that's how we stay decoupled from the eventual
backend and UI layers.
"""

from __future__ import annotations

import logging
import time
from threading import Event

from gpmodel.core.dispatcher import AlertDispatcher
from gpmodel.core.events import DetectionsReady, PerfSampled, StreamStateChanged
from gpmodel.core.interfaces import Detector, Tracker, VideoSource
from gpmodel.telemetry.perf import PerfMeter

logger = logging.getLogger(__name__)


class InferenceEngine:
    """Per-stream inference loop.

    One instance drives one stream — scaling to multiple streams is
    just multiple engines sharing a dispatcher (or running in separate
    threads/processes). The engine is cheap; the heavyweight state
    (models) lives in the detector.
    """

    def __init__(
        self,
        stream_id: str,
        source: VideoSource,
        detector: Detector,
        dispatcher: AlertDispatcher,
        tracker: Tracker | None = None,
        perf_window: int = 60,
        perf_emit_every: int = 30,
    ) -> None:
        self.stream_id = stream_id
        self.source = source
        self.detector = detector
        self.tracker = tracker
        self.dispatcher = dispatcher
        self._perf = PerfMeter(
            stream_id=stream_id, window=perf_window, emit_every=perf_emit_every
        )
        self._stop_event = Event()

    # ── Control ─────────────────────────────────────────────
    def stop(self) -> None:
        """Signal the run loop to exit after the current frame."""
        self._stop_event.set()

    @property
    def is_running(self) -> bool:
        return not self._stop_event.is_set()

    # ── Main loop ───────────────────────────────────────────
    def run(self) -> None:
        logger.info("Engine '%s' starting", self.stream_id)
        self.detector.warmup()

        try:
            self.source.open()
            self._emit_state("opened")

            for frame in self.source.frames():
                if self._stop_event.is_set():
                    logger.info("Engine '%s' stop requested", self.stream_id)
                    break

                t0 = time.perf_counter()
                detections = self.detector.detect(frame)
                tracks = self.tracker.update(detections, frame) if self.tracker else []
                latency_ms = (time.perf_counter() - t0) * 1000.0
                self._perf.tick(latency_ms)

                self.dispatcher.publish(
                    DetectionsReady(
                        stream_id=self.stream_id,
                        frame=frame,
                        detections=tuple(detections),
                        tracks=tuple(tracks),
                    )
                )

                if self._perf.should_emit():
                    self.dispatcher.publish(
                        PerfSampled(stream_id=self.stream_id, sample=self._perf.snapshot())
                    )

        except Exception as exc:
            logger.exception("Engine '%s' failed", self.stream_id)
            self._emit_state("error", str(exc))
            raise
        finally:
            self.source.close()
            self.detector.close()
            self._emit_state("closed")
            logger.info("Engine '%s' stopped", self.stream_id)

    # ── Internals ──────────────────────────────────────────
    def _emit_state(self, state: str, detail: str = "") -> None:
        self.dispatcher.publish(
            StreamStateChanged(stream_id=self.stream_id, state=state, detail=detail)
        )
