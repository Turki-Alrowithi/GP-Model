"""SAHI tiled detector — small-object detection via sliced inference.

Stock YOLO inference at 640² downsamples the frame so aggressively
that objects smaller than ~16 px (a knife in someone's hand at drone
altitude, a fence-breach from 40 m up) vanish. SAHI (Slicing Aided
Hyper Inference) tiles the full-resolution frame, runs the detector
on each tile, and fuses the results back into the global frame.

At 1080p with 640² tiles and 20% overlap that's typically 8 tiles
per frame — roughly 8x the compute but dramatically higher recall on
small targets. Use for drone/aerial footage, not for close-up webcam.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from gpmodel.core.types import BBox, Detection, Frame

logger = logging.getLogger(__name__)


class SahiYoloDetector:
    """YOLO detector wrapped in SAHI sliced inference.

    Same public API as YoloDetector (both satisfy the Detector
    Protocol), so the pipeline is blind to which one is wired in.
    """

    def __init__(
        self,
        weights: str | Path,
        device: str = "mps",
        confidence_threshold: float = 0.30,
        slice_height: int = 640,
        slice_width: int = 640,
        overlap_height_ratio: float = 0.20,
        overlap_width_ratio: float = 0.20,
        postprocess_type: str = "GREEDYNMM",
        postprocess_match_metric: str = "IOS",
        postprocess_match_threshold: float = 0.5,
        classes: Sequence[int] | None = None,
    ) -> None:
        from sahi import AutoDetectionModel  # heavy import — SAHI pulls in its own stack

        self.weights_path = Path(weights)
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.slice_height = slice_height
        self.slice_width = slice_width
        self.overlap_height_ratio = overlap_height_ratio
        self.overlap_width_ratio = overlap_width_ratio
        self.postprocess_type = postprocess_type
        self.postprocess_match_metric = postprocess_match_metric
        self.postprocess_match_threshold = postprocess_match_threshold
        self.classes = list(classes) if classes is not None else None

        logger.info("Loading SAHI+YOLO (%s) weights from %s", device, self.weights_path)
        self._model = AutoDetectionModel.from_pretrained(
            model_type="ultralytics",
            model_path=str(self.weights_path),
            confidence_threshold=self.confidence_threshold,
            device=self.device,
        )
        # SAHI's category_mapping keys are strings; normalise to int for
        # parity with YoloDetector.
        raw_mapping = dict(self._model.category_mapping or {})
        self.class_names: dict[int, str] = {int(k): v for k, v in raw_mapping.items()}

    # ── Lifecycle ───────────────────────────────────────────
    def warmup(self) -> None:
        """Run one sliced inference so tile caches / allocators are primed."""
        dummy = np.zeros((self.slice_height * 2, self.slice_width * 2, 3), dtype=np.uint8)
        self._predict(dummy)
        logger.info(
            "SAHI warmup complete (tile=%dx%d overlap=%.2f/%.2f)",
            self.slice_width,
            self.slice_height,
            self.overlap_width_ratio,
            self.overlap_height_ratio,
        )

    def close(self) -> None:
        """SAHI manages its own resources."""

    # ── Inference ───────────────────────────────────────────
    def detect(self, frame: Frame) -> list[Detection]:
        result = self._predict(frame.image)
        predictions: list[Any] = getattr(result, "object_prediction_list", [])
        out: list[Detection] = []
        for p in predictions:
            class_id = int(p.category.id)
            if self.classes is not None and class_id not in self.classes:
                continue
            bbox = p.bbox  # SAHI BoundingBox (minx, miny, maxx, maxy) via .to_xyxy()
            xyxy = (
                bbox.to_xyxy()
                if hasattr(bbox, "to_xyxy")
                else (
                    bbox.minx,
                    bbox.miny,
                    bbox.maxx,
                    bbox.maxy,
                )
            )
            out.append(
                Detection(
                    class_id=class_id,
                    class_name=self.class_names.get(class_id, str(class_id)),
                    confidence=float(p.score.value),
                    bbox=BBox(*(float(v) for v in xyxy)),
                )
            )
        return out

    # ── Internals ───────────────────────────────────────────
    def _predict(self, image: np.ndarray) -> Any:
        from sahi.predict import get_sliced_prediction

        return get_sliced_prediction(
            image=image,
            detection_model=self._model,
            slice_height=self.slice_height,
            slice_width=self.slice_width,
            overlap_height_ratio=self.overlap_height_ratio,
            overlap_width_ratio=self.overlap_width_ratio,
            postprocess_type=self.postprocess_type,
            postprocess_match_metric=self.postprocess_match_metric,
            postprocess_match_threshold=self.postprocess_match_threshold,
            verbose=0,
        )
