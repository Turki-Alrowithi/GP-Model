"""Shared base for OpenCV-backed video sources.

Concrete sources (webcam, file, RTSP) only need to describe *how* to
build the `cv2.VideoCapture` and whether to loop — everything else
(frame counting, timestamping, resource lifecycle) lives here.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Iterator
from datetime import UTC, datetime
from typing import Any

import cv2
import numpy as np

from gpmodel.core.types import Frame

logger = logging.getLogger(__name__)


class BaseVideoSource(ABC):
    """Template-method base for VideoSource implementations.

    Subclasses implement `_open_capture()` to construct the underlying
    capture object. The rest — open/close lifecycle, iteration, frame
    numbering — is handled here.
    """

    def __init__(self, stream_id: str) -> None:
        self.stream_id = stream_id
        self._capture: cv2.VideoCapture | None = None
        self._frame_count: int = 0

    # ── Lifecycle ───────────────────────────────────────────
    def open(self) -> None:
        if self._capture is not None:
            return
        cap = self._open_capture()
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video source '{self.stream_id}'")
        self._capture = cap
        logger.info("Opened source '%s' (%s)", self.stream_id, self._describe())

    def close(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    @property
    def is_open(self) -> bool:
        return self._capture is not None and self._capture.isOpened()

    # ── Iteration ───────────────────────────────────────────
    def frames(self) -> Iterator[Frame]:
        if not self.is_open:
            self.open()
        assert self._capture is not None  # narrow for type checker

        while True:
            ok, image = self._capture.read()
            if not ok or image is None:
                if self._should_reopen_on_eof():
                    self._rewind()
                    continue
                break
            yield self._make_frame(image)

    def __iter__(self) -> Iterator[Frame]:
        return self.frames()

    def __enter__(self) -> BaseVideoSource:
        self.open()
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    # ── Subclass hooks ──────────────────────────────────────
    @abstractmethod
    def _open_capture(self) -> cv2.VideoCapture:
        """Construct and return the underlying VideoCapture."""

    def _should_reopen_on_eof(self) -> bool:
        """Override to enable looping (e.g. file source with loop=True)."""
        return False

    def _describe(self) -> str:
        """Short string for logs."""
        return type(self).__name__

    # ── Internals ───────────────────────────────────────────
    def _rewind(self) -> None:
        if self._capture is not None:
            self._capture.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def _make_frame(self, image: np.ndarray) -> Frame:
        self._frame_count += 1
        return Frame(
            stream_id=self.stream_id,
            frame_id=self._frame_count,
            timestamp=datetime.now(UTC),
            image=image,
        )
