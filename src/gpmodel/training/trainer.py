"""Thin, typed wrapper over Ultralytics' training loop.

Just enough to keep our hand on the wheel: a fixed input contract, a
predictable output directory, and a small result object with the
path of the best weights — no magic strings flying around the rest
of the codebase.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class TrainResult:
    best_weights: Path
    last_weights: Path
    run_dir: Path
    data: Path
    base_weights: Path
    epochs: int
    imgsz: int


def train(
    base_weights: str | Path,
    data: str | Path,
    epochs: int = 50,
    imgsz: int = 640,
    batch: int = 16,
    device: str = "mps",
    project: str | Path = Path("runs/train"),
    name: str | None = None,
    resume: bool = False,
    patience: int = 20,
    lr0: float | None = None,
    workers: int = 4,
    seed: int = 0,
) -> TrainResult:
    """Fine-tune a YOLO model on a custom dataset.

    `data` is an Ultralytics-format dataset YAML (see
    `datasets/example_data.yaml` for the expected shape).

    The run lands in `project/name/`. Best weights are at
    `project/name/weights/best.pt` and are returned in `TrainResult`.
    """
    from ultralytics import YOLO  # type: ignore[attr-defined]  # heavy import

    base = Path(base_weights)
    data_path = Path(data)
    if not base.exists():
        raise FileNotFoundError(f"Base weights not found: {base}")
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {data_path}")

    logger.info(
        "Training %s on %s for %d epochs (imgsz=%d batch=%d device=%s)",
        base.name,
        data_path.name,
        epochs,
        imgsz,
        batch,
        device,
    )

    model = YOLO(str(base))
    overrides: dict[str, object] = {
        "data": str(data_path),
        "epochs": epochs,
        "imgsz": imgsz,
        "batch": batch,
        "device": device,
        "project": str(project),
        "workers": workers,
        "patience": patience,
        "seed": seed,
        "resume": resume,
    }
    if name is not None:
        overrides["name"] = name
    if lr0 is not None:
        overrides["lr0"] = lr0

    results = model.train(**overrides)

    # Ultralytics Trainer object → save_dir
    fallback_dir = Path(project) / (name or "exp")
    save_dir = Path(getattr(results, "save_dir", fallback_dir))
    best = save_dir / "weights" / "best.pt"
    last = save_dir / "weights" / "last.pt"
    if not best.exists():
        raise RuntimeError(f"Training reported success but best.pt missing in {save_dir}")

    logger.info("Training complete. Best weights: %s", best)
    return TrainResult(
        best_weights=best,
        last_weights=last,
        run_dir=save_dir,
        data=data_path,
        base_weights=base,
        epochs=epochs,
        imgsz=imgsz,
    )
