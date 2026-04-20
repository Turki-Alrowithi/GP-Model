"""Video sources — webcam, file, and (later) RTSP/RTMP streams."""

from gpmodel.sources.base import BaseVideoSource
from gpmodel.sources.file import FileSource
from gpmodel.sources.webcam import WebcamSource

__all__ = ["BaseVideoSource", "FileSource", "WebcamSource"]
