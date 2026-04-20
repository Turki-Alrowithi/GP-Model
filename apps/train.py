#!/usr/bin/env python3
"""CLI: fine-tune a YOLO model on a custom dataset.

Designed to run both locally (MPS / CPU) and on Colab (CUDA). Example:

    # Local, M-series GPU
    uv run python apps/train.py \\
        --weights yolo11s.pt \\
        --data datasets/weapons.yaml \\
        --epochs 50 --imgsz 640 --device mps

    # Colab, T4/A100
    !python apps/train.py --weights yolo11s.pt \\
        --data datasets/weapons.yaml \\
        --epochs 100 --imgsz 640 --batch 32 --device 0

The training run lands in `runs/train/<name>/`, with the best weights
at `runs/train/<name>/weights/best.pt`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from gpmodel.telemetry.logging import configure_logging
from gpmodel.training.trainer import train


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="gpmodel-train",
        description="Fine-tune a YOLO model on a custom dataset.",
    )
    p.add_argument(
        "--weights",
        "-w",
        type=Path,
        default=Path("yolo11s.pt"),
        help="Base checkpoint to fine-tune from (default: yolo11s.pt)",
    )
    p.add_argument("--data", "-d", type=Path, required=True, help="Ultralytics-format dataset YAML")
    p.add_argument("--epochs", "-e", type=int, default=50)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", "-b", type=int, default=16)
    p.add_argument("--device", default="mps", help="mps | cpu | cuda | 0 | 0,1 (default: mps)")
    p.add_argument("--project", type=Path, default=Path("runs/train"))
    p.add_argument("--name", default=None, help="Run name (default: autoincrement)")
    p.add_argument(
        "--resume",
        action="store_true",
        help="Resume from --weights (expects a last.pt from a prior run)",
    )
    p.add_argument("--patience", type=int, default=20, help="Early stopping patience")
    p.add_argument("--lr0", type=float, default=None, help="Initial learning rate override")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_logging(level="INFO", fmt="console")

    result = train(
        base_weights=args.weights,
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        resume=args.resume,
        patience=args.patience,
        lr0=args.lr0,
        workers=args.workers,
        seed=args.seed,
    )

    print("\nTraining complete.")
    print(f"  Run dir:      {result.run_dir}")
    print(f"  Best:         {result.best_weights}")
    print(f"  Last:         {result.last_weights}")
    print("\nNext steps:")
    print(f"  uv run python apps/eval.py --weights {result.best_weights} --data {result.data}")
    print(f"  uv run python apps/export.py export --weights {result.best_weights} --format onnx")
    return 0


if __name__ == "__main__":
    sys.exit(main())
