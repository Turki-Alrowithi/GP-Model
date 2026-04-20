"""Pydantic config schema — single source of truth for run parameters.

Loaded from YAML, validated at startup. Every concrete component
(sources, detector, tracker, publishers) has its own typed section;
unknown keys are rejected so typos surface immediately.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


# ── Source ─────────────────────────────────────────────────
class WebcamConfig(_Strict):
    type: Literal["webcam"] = "webcam"
    device_index: int = 0
    width: int | None = 1920
    height: int | None = 1080
    fps: float | None = 30.0


class FileConfig(_Strict):
    type: Literal["file"] = "file"
    path: Path
    loop: bool = False


class RtspConfig(_Strict):
    type: Literal["rtsp"] = "rtsp"
    url: str
    transport: Literal["tcp", "udp"] = "tcp"
    reconnect_delay_s: float = 1.0
    max_reconnect_delay_s: float = 30.0


SourceConfig = WebcamConfig | FileConfig | RtspConfig


# ── Detector ───────────────────────────────────────────────
class YoloConfig(_Strict):
    type: Literal["yolo"] = "yolo"
    weights: str = "yolo11s.pt"
    device: Literal["mps", "cpu", "cuda"] = "mps"
    imgsz: int = 640
    conf: float = Field(default=0.30, ge=0.0, le=1.0)
    iou: float = Field(default=0.45, ge=0.0, le=1.0)
    classes: list[int] | None = None
    half: bool = False


DetectorConfig = YoloConfig


# ── Tracker ────────────────────────────────────────────────
class ByteTrackConfig(_Strict):
    enabled: bool = True
    type: Literal["bytetrack"] = "bytetrack"
    fps: float = 30.0
    track_activation_threshold: float = Field(default=0.25, ge=0.0, le=1.0)
    lost_track_buffer: int = Field(default=30, ge=0)
    minimum_matching_threshold: float = Field(default=0.8, ge=0.0, le=1.0)
    minimum_consecutive_frames: int = Field(default=1, ge=1)


TrackerConfig = ByteTrackConfig


# ── Publishers ─────────────────────────────────────────────
class ConsolePublisherConfig(_Strict):
    enabled: bool = True
    print_detections: bool = False


class JSONLPublisherConfig(_Strict):
    enabled: bool = False
    path: Path = Path("logs/events.jsonl")
    include_detection_frames: bool = False


class MetricsPublisherConfig(_Strict):
    enabled: bool = True


class PublishersConfig(_Strict):
    console: ConsolePublisherConfig = ConsolePublisherConfig()
    jsonl: JSONLPublisherConfig = JSONLPublisherConfig()
    metrics: MetricsPublisherConfig = MetricsPublisherConfig()


# ── Rules ──────────────────────────────────────────────────
class GeofenceZoneConfig(_Strict):
    name: str
    # List of (x, y) vertices. Normalized (0-1) by default; set
    # normalized: false to supply absolute pixel coordinates.
    points: list[tuple[float, float]] = Field(min_length=3)
    normalized: bool = True


class GeofenceRuleConfig(_Strict):
    enabled: bool = True
    zones: list[GeofenceZoneConfig] = Field(default_factory=list)
    classes: list[str] = Field(default_factory=lambda: ["person"])
    severity: Literal["LOW", "MEDIUM", "HIGH", "CRITICAL", "OPERATIONAL"] = "HIGH"
    cooldown_s: float = Field(default=30.0, ge=0.0)
    foot_point: bool = True


class RulesConfig(_Strict):
    geofence: GeofenceRuleConfig = GeofenceRuleConfig()


# ── Performance & logging ──────────────────────────────────
class PerfConfig(_Strict):
    window: int = Field(default=60, ge=1)
    emit_every: int = Field(default=30, ge=1)


class LoggingConfig(_Strict):
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    format: Literal["console", "json"] = "console"


# ── Stream (one engine) ────────────────────────────────────
class StreamConfig(_Strict):
    id: str = "stream-01"
    source: SourceConfig = Field(discriminator="type", default=WebcamConfig())


# ── Top-level ──────────────────────────────────────────────
class AppConfig(_Strict):
    stream: StreamConfig = StreamConfig()
    detector: DetectorConfig = YoloConfig()
    tracker: TrackerConfig = ByteTrackConfig()
    rules: RulesConfig = RulesConfig()
    publishers: PublishersConfig = PublishersConfig()
    perf: PerfConfig = PerfConfig()
    logging: LoggingConfig = LoggingConfig()
