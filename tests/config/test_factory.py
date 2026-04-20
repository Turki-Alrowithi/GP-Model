"""Tests for the component factory — build from validated config."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from gpmodel.config.factory import (
    build_publishers,
    build_source,
    build_tracker,
)
from gpmodel.config.schema import (
    ByteTrackConfig,
    ConsolePublisherConfig,
    FileConfig,
    JSONLPublisherConfig,
    MetricsPublisherConfig,
    PublishersConfig,
    RtspConfig,
    SahiYoloConfig,
    WebcamConfig,
)
from gpmodel.publishers.metrics import MetricsSubscriber
from gpmodel.sources.file import FileSource
from gpmodel.sources.rtsp import RtspSource
from gpmodel.sources.webcam import WebcamSource


def test_build_webcam_source() -> None:
    cfg = WebcamConfig(device_index=2, width=1280, height=720, fps=60)
    src = build_source(cfg, "cam-A")
    assert isinstance(src, WebcamSource)
    assert src.stream_id == "cam-A"
    assert src.device_index == 2
    assert src.width == 1280


def test_build_file_source(tmp_path: Path) -> None:
    video = tmp_path / "v.mp4"
    video.touch()
    cfg = FileConfig(path=video, loop=True)

    src = build_source(cfg, "file-A")
    assert isinstance(src, FileSource)
    assert src.loop


def test_build_rtsp_source() -> None:
    cfg = RtspConfig(url="rtsp://cam-07:8554/live", transport="udp")
    src = build_source(cfg, "drone-07")
    assert isinstance(src, RtspSource)
    assert src.url == "rtsp://cam-07:8554/live"
    assert src.transport == "udp"
    assert src.stream_id == "drone-07"


def test_build_tracker_respects_enabled_flag() -> None:
    assert build_tracker(ByteTrackConfig(enabled=False)) is None
    assert build_tracker(ByteTrackConfig(enabled=True)) is not None


@patch("ultralytics.YOLO")
def test_build_detector_forwards_yolo_args(yolo_cls: MagicMock) -> None:
    yolo_cls.return_value = MagicMock(names={})
    from gpmodel.config.factory import build_detector
    from gpmodel.config.schema import YoloConfig

    cfg = YoloConfig(weights="yolo11s.pt", device="cpu", imgsz=320, conf=0.4)
    det = build_detector(cfg)
    assert det.device == "cpu"
    assert det.imgsz == 320


@patch("sahi.AutoDetectionModel")
def test_build_detector_dispatches_to_sahi(auto_cls: MagicMock) -> None:
    auto_cls.from_pretrained.return_value = MagicMock(category_mapping={})
    from gpmodel.config.factory import build_detector
    from gpmodel.detectors.sahi import SahiYoloDetector

    cfg = SahiYoloConfig(
        weights="yolo11s.pt",
        device="cpu",
        slice_height=512,
        slice_width=512,
        overlap_height_ratio=0.25,
    )
    det = build_detector(cfg)
    assert isinstance(det, SahiYoloDetector)
    assert det.slice_height == 512
    assert det.overlap_height_ratio == 0.25


def test_build_publishers_only_enabled_ones() -> None:
    cfg = PublishersConfig(
        console=ConsolePublisherConfig(enabled=False),
        jsonl=JSONLPublisherConfig(enabled=False),
        metrics=MetricsPublisherConfig(enabled=True),
    )
    subs, metrics = build_publishers(cfg)
    assert len(subs) == 1
    assert isinstance(metrics, MetricsSubscriber)


def test_build_publishers_all_enabled(tmp_path: Path) -> None:
    log = tmp_path / "events.jsonl"
    cfg = PublishersConfig(
        console=ConsolePublisherConfig(enabled=True),
        jsonl=JSONLPublisherConfig(enabled=True, path=log),
        metrics=MetricsPublisherConfig(enabled=True),
    )
    subs, metrics = build_publishers(cfg)
    assert len(subs) == 3
    assert metrics is not None
