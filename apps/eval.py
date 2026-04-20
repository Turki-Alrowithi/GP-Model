#!/usr/bin/env python3
"""CLI: evaluate a YOLO model on a validation split and report mAP.

    uv run python apps/eval.py \\
        --weights runs/train/weapons/weights/best.pt \\
        --data datasets/weapons.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from gpmodel.telemetry.logging import configure_logging
from gpmodel.training.evaluator import evaluate


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="gpmodel-eval",
        description="Evaluate a YOLO model and print mAP / precision / recall.",
    )
    p.add_argument("--weights", "-w", type=Path, required=True)
    p.add_argument("--data", "-d", type=Path, required=True)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", "-b", type=int, default=16)
    p.add_argument("--device", default="mps")
    p.add_argument("--split", default="val", choices=["val", "test"])
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_logging(level="INFO", fmt="console")

    result = evaluate(
        weights=args.weights,
        data=args.data,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        split=args.split,
    )
    print(result.pretty())
    return 0


if __name__ == "__main__":
    sys.exit(main())
