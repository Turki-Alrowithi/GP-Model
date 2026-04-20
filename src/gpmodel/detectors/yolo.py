"""YOLO detector — Ultralytics backend with Apple Silicon acceleration.

One class handles every deployment format Ultralytics understands:

- `.pt`       → PyTorch (on `mps` for M-series GPU, or `cpu`)
- `.mlpackage`→ Apple CoreML (Neural Engine + GPU)
- `.onnx`     → ONNX Runtime
- `.engine`   → TensorRT (NVIDIA only; unused on M-series)

The file extension dictates the runtime; the `device` argument only
applies to PyTorch loads.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from gpmodel.core.types import BBox, Detection, Frame

logger = logging.getLogger(__name__)


class YoloDetector:
    """Thin, typed wrapper over `ultralytics.YOLO`.

    Performs per-frame inference and returns our domain `Detection`
    objects — never raw tensors — so the rest of the pipeline is
    framework-agnostic.
    """

    def __init__(
        self,
        weights: str | Path,
        device: str = "mps",
        imgsz: int = 640,
        conf: float = 0.30,
        iou: float = 0.45,
        classes: Sequence[int] | None = None,
        half: bool = False,
    ) -> None:
        from ultralytics import YOLO  # type: ignore[attr-defined]  # local import — heavy module

        self.weights_path = Path(weights)
        self.device = device
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
        self.classes = list(classes) if classes is not None else None
        self.half = half

        logger.info("Loading YOLO weights from %s", self.weights_path)
        self._model = YOLO(str(self.weights_path))
        # names is a dict[int, str]; copy to decouple from the upstream object
        self.class_names: dict[int, str] = dict(self._model.names)

    # ── Lifecycle ───────────────────────────────────────────
    def warmup(self) -> None:
        """Run a dummy inference so the first real frame pays no compile cost."""
        dummy = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
        self._predict(dummy)
        logger.info(
            "YOLO warmup complete (device=%s imgsz=%d conf=%.2f)",
            self.device,
            self.imgsz,
            self.conf,
        )

    def close(self) -> None:
        """No-op — Ultralytics manages its own resources."""

    # ── Inference ───────────────────────────────────────────
    def detect(self, frame: Frame) -> list[Detection]:
        results = self._predict(frame.image)
        if not results:
            return []
        return list(self._to_detections(results[0]))

    # ── Internals ───────────────────────────────────────────
    def _predict(self, image: np.ndarray) -> Sequence[Any]:
        return self._model.predict(
            source=image,
            device=self.device,
            imgsz=self.imgsz,
            conf=self.conf,
            iou=self.iou,
            classes=self.classes,
            half=self.half,
            verbose=False,
        )

    def _to_detections(self, result: object) -> list[Detection]:
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return []

        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        cls_ids = boxes.cls.cpu().numpy().astype(int)

        out: list[Detection] = []
        for (x1, y1, x2, y2), cf, cid in zip(xyxy, confs, cls_ids, strict=False):
            out.append(
                Detection(
                    class_id=int(cid),
                    class_name=self.class_names.get(int(cid), str(cid)),
                    confidence=float(cf),
                    bbox=BBox(float(x1), float(y1), float(x2), float(y2)),
                )
            )
        return out
