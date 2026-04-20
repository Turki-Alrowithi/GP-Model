"""Core domain types — frames, detections, tracks, bounding boxes.

These are the value objects that flow through the inference pipeline.
Keep them immutable where practical and free of behavior beyond simple
derivations; business logic lives in the pipeline stages and rules.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import numpy as np


# ── Bounding box ────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class BBox:
    """Axis-aligned bounding box in absolute pixel coordinates (xyxy)."""

    x1: float
    y1: float
    x2: float
    y2: float

    def __post_init__(self) -> None:
        if self.x2 < self.x1 or self.y2 < self.y1:
            raise ValueError(f"Invalid BBox: ({self.x1},{self.y1})-({self.x2},{self.y2})")

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    def as_xyxy(self) -> tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)

    def as_xywh(self) -> tuple[float, float, float, float]:
        return (self.x1, self.y1, self.width, self.height)

    def iou(self, other: BBox) -> float:
        """Intersection-over-union with another box."""
        ix1 = max(self.x1, other.x1)
        iy1 = max(self.y1, other.y1)
        ix2 = min(self.x2, other.x2)
        iy2 = min(self.y2, other.y2)
        iw = max(0.0, ix2 - ix1)
        ih = max(0.0, iy2 - iy1)
        inter = iw * ih
        union = self.area + other.area - inter
        return inter / union if union > 0 else 0.0


# ── Detection ──────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class Detection:
    """A single object detection output by a Detector."""

    class_id: int
    class_name: str
    confidence: float
    bbox: BBox
    track_id: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)


# ── Track ──────────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class Track:
    """A confirmed track over multiple frames (output by a Tracker)."""

    track_id: int
    class_id: int
    class_name: str
    confidence: float
    bbox: BBox
    age: int  # number of updates since first seen
    time_since_update: int  # frames since last matched detection


# ── Frame ──────────────────────────────────────────────────
@dataclass(slots=True)
class Frame:
    """A single frame of video plus its metadata.

    The image is held by reference (no copy) — downstream stages must
    treat it as read-only unless they explicitly copy.
    """

    stream_id: str
    frame_id: int
    timestamp: datetime  # UTC
    image: np.ndarray  # BGR, shape (H, W, 3)

    @property
    def width(self) -> int:
        return int(self.image.shape[1])

    @property
    def height(self) -> int:
        return int(self.image.shape[0])

    @property
    def shape(self) -> tuple[int, int]:
        return (self.height, self.width)


# ── Performance sample ─────────────────────────────────────
@dataclass(frozen=True, slots=True)
class PerfSample:
    """Periodic performance snapshot for observability."""

    stream_id: str
    fps: float
    latency_ms: float
    frame_count: int
    dropped_frames: int
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
