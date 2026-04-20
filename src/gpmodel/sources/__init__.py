"""Video sources — webcam, file, RTSP, and a threaded reader wrapper."""

from gpmodel.sources.base import BaseVideoSource
from gpmodel.sources.file import FileSource
from gpmodel.sources.rtsp import RtspSource
from gpmodel.sources.threaded import ThreadedFrameReader
from gpmodel.sources.webcam import WebcamSource

__all__ = [
    "BaseVideoSource",
    "FileSource",
    "RtspSource",
    "ThreadedFrameReader",
    "WebcamSource",
]
