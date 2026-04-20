"""Webcam video source — wraps a local V4L / AVFoundation camera via OpenCV."""

from __future__ import annotations

import cv2

from gpmodel.sources.base import BaseVideoSource


class WebcamSource(BaseVideoSource):
    """Integrated or USB webcam.

    On macOS OpenCV uses AVFoundation under the hood; on Linux V4L2.
    Resolution/FPS hints are advisory — the driver may ignore them
    and deliver whatever it can.
    """

    def __init__(
        self,
        stream_id: str = "webcam-0",
        device_index: int = 0,
        width: int | None = 1920,
        height: int | None = 1080,
        fps: float | None = 30.0,
    ) -> None:
        super().__init__(stream_id)
        self.device_index = device_index
        self.width = width
        self.height = height
        self.fps = fps

    def _open_capture(self) -> cv2.VideoCapture:
        cap = cv2.VideoCapture(self.device_index)
        if self.width is not None:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(self.width))
        if self.height is not None:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(self.height))
        if self.fps is not None:
            cap.set(cv2.CAP_PROP_FPS, float(self.fps))
        return cap

    def _describe(self) -> str:
        return f"webcam index={self.device_index} req={self.width}x{self.height}@{self.fps}"
