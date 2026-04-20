"""File video source — read a pre-recorded clip as if it were a live stream."""

from __future__ import annotations

from pathlib import Path

import cv2

from gpmodel.sources.base import BaseVideoSource


class FileSource(BaseVideoSource):
    """Play a video file once (or in a loop) as a drone stream.

    Useful for deterministic demos and regression tests — pair with a
    fixed-size sample clip and the pipeline behaves identically on
    every run.
    """

    def __init__(
        self,
        path: str | Path,
        stream_id: str | None = None,
        loop: bool = False,
    ) -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"Video file not found: {self.path}")
        super().__init__(stream_id or f"file:{self.path.name}")
        self.loop = loop

    def _open_capture(self) -> cv2.VideoCapture:
        return cv2.VideoCapture(str(self.path))

    def _should_reopen_on_eof(self) -> bool:
        return self.loop

    def _describe(self) -> str:
        return f"file={self.path.name} loop={self.loop}"
