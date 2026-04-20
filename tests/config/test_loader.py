"""Tests for the YAML config loader + schema validation."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from gpmodel.config.loader import load_config
from gpmodel.config.schema import AppConfig, FileConfig, RtspConfig, WebcamConfig


def test_defaults_when_file_is_empty(tmp_path: Path) -> None:
    path = tmp_path / "empty.yaml"
    path.write_text("")

    cfg = load_config(path)
    assert isinstance(cfg, AppConfig)
    assert isinstance(cfg.stream.source, WebcamConfig)
    assert cfg.detector.weights == "yolo11s.pt"
    assert cfg.tracker.enabled


def test_loads_webcam_config(tmp_path: Path) -> None:
    path = tmp_path / "c.yaml"
    path.write_text(
        """
stream:
  id: "cam-1"
  source:
    type: webcam
    device_index: 1
    width: 1280
    height: 720
    fps: 60
detector:
  weights: yolo11n.pt
  device: cpu
  conf: 0.5
"""
    )
    cfg = load_config(path)
    assert cfg.stream.id == "cam-1"
    assert isinstance(cfg.stream.source, WebcamConfig)
    assert cfg.stream.source.device_index == 1
    assert cfg.detector.device == "cpu"
    assert cfg.detector.conf == 0.5


def test_loads_file_source_config(tmp_path: Path) -> None:
    video = tmp_path / "vid.mp4"
    video.touch()
    path = tmp_path / "c.yaml"
    path.write_text(
        f"""
stream:
  id: "file-1"
  source:
    type: file
    path: {video}
    loop: true
"""
    )
    cfg = load_config(path)
    assert isinstance(cfg.stream.source, FileConfig)
    assert cfg.stream.source.loop


def test_loads_rtsp_source_config(tmp_path: Path) -> None:
    path = tmp_path / "c.yaml"
    path.write_text(
        """
stream:
  id: "drone-1"
  source:
    type: rtsp
    url: rtsp://localhost:8554/drone01
    transport: udp
    reconnect_delay_s: 0.5
"""
    )
    cfg = load_config(path)
    assert isinstance(cfg.stream.source, RtspConfig)
    assert cfg.stream.source.url == "rtsp://localhost:8554/drone01"
    assert cfg.stream.source.transport == "udp"
    assert cfg.stream.source.reconnect_delay_s == 0.5


def test_rejects_unknown_keys(tmp_path: Path) -> None:
    path = tmp_path / "c.yaml"
    path.write_text(
        """
detector:
  weights: yolo11n.pt
  mystery_field: true
"""
    )
    with pytest.raises(ValidationError):
        load_config(path)


def test_rejects_out_of_range_conf(tmp_path: Path) -> None:
    path = tmp_path / "c.yaml"
    path.write_text("detector:\n  conf: 1.5\n")
    with pytest.raises(ValidationError):
        load_config(path)


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "missing.yaml")
