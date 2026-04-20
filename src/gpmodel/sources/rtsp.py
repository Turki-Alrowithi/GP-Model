"""RTSP video source — consumes a live RTSP stream via OpenCV/FFmpeg.

Pairs with the MediaMTX + ffmpeg compose stack under `docker/` so the
inference engine can run against a *real* network stream locally, with
no drone hardware attached.

Reconnection is handled automatically: if the network blips or the
publisher restarts, the source re-opens the underlying VideoCapture.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator

import cv2

from gpmodel.core.types import Frame
from gpmodel.sources.base import BaseVideoSource

logger = logging.getLogger(__name__)


class RtspSource(BaseVideoSource):
    """An RTSP stream reader with automatic reconnection.

    Uses FFmpeg via OpenCV under the hood. Transport defaults to TCP
    (more reliable than UDP for unreliable networks); override via
    `transport="udp"` for lower latency on a clean LAN.
    """

    def __init__(
        self,
        url: str,
        stream_id: str | None = None,
        transport: str = "tcp",
        reconnect_delay_s: float = 1.0,
        max_reconnect_delay_s: float = 30.0,
    ) -> None:
        super().__init__(stream_id or f"rtsp:{url}")
        self.url = url
        self.transport = transport
        self.reconnect_delay_s = reconnect_delay_s
        self.max_reconnect_delay_s = max_reconnect_delay_s

    # ── Subclass hooks ──────────────────────────────────────
    def _open_capture(self) -> cv2.VideoCapture:
        # Hinting FFmpeg about transport via environment avoids the UDP
        # default, which drops a lot of frames on congested networks.
        import os

        os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", f"rtsp_transport;{self.transport}")
        return cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)

    def _describe(self) -> str:
        return f"rtsp url={self.url} transport={self.transport}"

    # ── Override iteration with reconnect loop ──────────────
    def frames(self) -> Iterator[Frame]:
        if not self.is_open:
            self.open()
        delay = self.reconnect_delay_s

        while True:
            assert self._capture is not None
            ok, image = self._capture.read()
            if ok and image is not None:
                delay = self.reconnect_delay_s  # reset backoff on good frame
                yield self._make_frame(image)
                continue

            # Lost the stream — release and reconnect with backoff.
            logger.warning("RTSP '%s' read failed; reconnecting in %.1fs", self.stream_id, delay)
            self._capture.release()
            self._capture = None
            time.sleep(delay)
            delay = min(delay * 2, self.max_reconnect_delay_s)
            try:
                self.open()
            except RuntimeError:
                logger.warning("RTSP '%s' reconnect failed; will retry", self.stream_id)
