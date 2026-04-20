"""Unit tests for RtspSource.

A real RTSP round-trip is an integration concern (requires the
MediaMTX compose stack to be running); these tests exercise the
construction and FFmpeg transport plumbing without hitting the network.
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

from gpmodel.core.interfaces import VideoSource
from gpmodel.sources.rtsp import RtspSource


def test_is_a_video_source() -> None:
    src = RtspSource(url="rtsp://example/stream")
    assert isinstance(src, VideoSource)


def test_default_stream_id_uses_url() -> None:
    src = RtspSource(url="rtsp://example/stream")
    assert src.stream_id == "rtsp:rtsp://example/stream"


def test_custom_stream_id_overrides_default() -> None:
    src = RtspSource(url="rtsp://x", stream_id="drone07")
    assert src.stream_id == "drone07"


@patch("cv2.VideoCapture")
def test_open_sets_ffmpeg_transport_env(cap_cls: MagicMock) -> None:
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    cap_cls.return_value = mock_cap

    # Reset the env var before the test so defaults are predictable.
    os.environ.pop("OPENCV_FFMPEG_CAPTURE_OPTIONS", None)

    src = RtspSource(url="rtsp://x", transport="tcp")
    src.open()

    assert "rtsp_transport;tcp" in os.environ.get("OPENCV_FFMPEG_CAPTURE_OPTIONS", "")


@patch("cv2.VideoCapture")
def test_open_accepts_udp_transport(cap_cls: MagicMock) -> None:
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    cap_cls.return_value = mock_cap

    os.environ.pop("OPENCV_FFMPEG_CAPTURE_OPTIONS", None)
    RtspSource(url="rtsp://x", transport="udp").open()

    assert "rtsp_transport;udp" in os.environ.get("OPENCV_FFMPEG_CAPTURE_OPTIONS", "")


@patch("cv2.VideoCapture")
def test_raises_when_capture_fails_to_open(cap_cls: MagicMock) -> None:
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = False
    cap_cls.return_value = mock_cap

    src = RtspSource(url="rtsp://down")
    import pytest

    with pytest.raises(RuntimeError, match="Failed to open"):
        src.open()
