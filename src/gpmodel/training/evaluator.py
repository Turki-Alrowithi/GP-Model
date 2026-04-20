"""Thin, typed wrapper over Ultralytics' `.val()` for mAP reporting."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class EvalResult:
    weights: Path
    data: Path
    imgsz: int
    map50: float
    map50_95: float
    precision: float
    recall: float

    def pretty(self) -> str:
        return (
            f"\n  weights:    {self.weights}"
            f"\n  data:       {self.data}"
            f"\n  imgsz:      {self.imgsz}"
            f"\n  mAP@50:     {self.map50:.3f}"
            f"\n  mAP@50-95:  {self.map50_95:.3f}"
            f"\n  precision:  {self.precision:.3f}"
            f"\n  recall:     {self.recall:.3f}\n"
        )


def evaluate(
    weights: str | Path,
    data: str | Path,
    imgsz: int = 640,
    batch: int = 16,
    device: str = "mps",
    split: str = "val",
) -> EvalResult:
    """Run validation on a YOLO model and collect mAP + PR metrics."""
    from ultralytics import YOLO  # type: ignore[attr-defined]

    w = Path(weights)
    d = Path(data)
    if not w.exists():
        raise FileNotFoundError(f"Weights not found: {w}")
    if not d.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {d}")

    logger.info("Evaluating %s on %s (%s split, imgsz=%d)", w.name, d.name, split, imgsz)
    model = YOLO(str(w))
    metrics = model.val(
        data=str(d),
        imgsz=imgsz,
        batch=batch,
        device=device,
        split=split,
        verbose=False,
    )
    box = metrics.box

    return EvalResult(
        weights=w,
        data=d,
        imgsz=imgsz,
        map50=float(getattr(box, "map50", 0.0)),
        map50_95=float(getattr(box, "map", 0.0)),
        precision=float(getattr(box, "mp", 0.0)),
        recall=float(getattr(box, "mr", 0.0)),
    )
