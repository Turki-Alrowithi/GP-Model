"""Abstract interfaces for swappable components (Strategy pattern).

Any concrete `VideoSource`, `Detector`, `Tracker`, or `Subscriber` implements
the matching Protocol. The pipeline depends on these protocols only —
never on concrete classes — so new backends can be added without
refactoring upstream code.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Protocol, runtime_checkable

from gpmodel.core.events import Event
from gpmodel.core.types import Detection, Frame, Track


@runtime_checkable
class VideoSource(Protocol):
    """Produces a stream of frames (webcam, file, RTSP, RTMP, ...)."""

    stream_id: str

    def open(self) -> None:
        """Acquire the underlying resource (camera handle, network socket, ...)."""

    def close(self) -> None:
        """Release the resource. Must be idempotent."""

    def frames(self) -> Iterator[Frame]:
        """Yield frames until exhaustion or `close()`."""

    @property
    def is_open(self) -> bool: ...


@runtime_checkable
class Detector(Protocol):
    """Object detector (YOLO family, DETR, custom, ...)."""

    def warmup(self) -> None:
        """Run a dummy inference so the first real frame isn't slow."""

    def detect(self, frame: Frame) -> list[Detection]:
        """Return detections for a single frame."""

    def close(self) -> None:
        """Free any allocated resources."""


@runtime_checkable
class Tracker(Protocol):
    """Multi-object tracker (ByteTrack, BoT-SORT, StrongSORT, ...)."""

    def update(self, detections: list[Detection], frame: Frame) -> list[Track]:
        """Update internal state with new detections and return confirmed tracks."""

    def reset(self) -> None:
        """Clear all tracks."""


@runtime_checkable
class Subscriber(Protocol):
    """Observer — handles events published by the pipeline."""

    def on_event(self, event: Event) -> None: ...
