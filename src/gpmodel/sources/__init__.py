"""Video sources — webcam, file, threaded reader, and (later) RTSP/RTMP."""

from gpmodel.sources.base import BaseVideoSource
from gpmodel.sources.file import FileSource
from gpmodel.sources.threaded import ThreadedFrameReader
from gpmodel.sources.webcam import WebcamSource

__all__ = ["BaseVideoSource", "FileSource", "ThreadedFrameReader", "WebcamSource"]
