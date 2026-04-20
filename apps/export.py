#!/usr/bin/env python3
"""CLI: export YOLO weights and benchmark detectors.

Examples:

    # Convert stock YOLO11s PyTorch weights to Apple CoreML.
    uv run python apps/export.py export \\
        --weights yolo11s.pt --format coreml

    # Benchmark the resulting .mlpackage on the bundled sample clip.
    uv run python apps/export.py bench \\
        --weights models/yolo11s.mlpackage \\
        --source assets/samples/drone_01.mp4 \\
        --frames 200

    # Benchmark the original PyTorch weights on MPS for comparison.
    uv run python apps/export.py bench \\
        --weights yolo11s.pt --device mps \\
        --source assets/samples/drone_01.mp4
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from gpmodel.export.benchmark import benchmark
from gpmodel.export.exporter import export_model
from gpmodel.telemetry.logging import configure_logging

logger = logging.getLogger(__name__)


def _cmd_export(args: argparse.Namespace) -> int:
    result = export_model(
        weights=args.weights,
        fmt=args.format,
        output_dir=args.output,
        imgsz=args.imgsz,
        half=args.half,
        nms=not args.no_nms,
    )
    print(f"\nExported: {result.path}\n")
    return 0


def _cmd_bench(args: argparse.Namespace) -> int:
    result = benchmark(
        weights=args.weights,
        source=args.source,
        device=args.device,
        imgsz=args.imgsz,
        conf=args.conf,
        n_frames=args.frames,
        warmup=args.warmup,
    )
    print(result.pretty())
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gpmodel-export",
        description="Export and benchmark detector weights.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ── export ────────────────────────────────────────────────
    pe = sub.add_parser("export", help="Convert PyTorch weights to CoreML/ONNX/TorchScript.")
    pe.add_argument("--weights", "-w", required=True, type=Path, help="Source .pt file")
    pe.add_argument(
        "--format",
        "-f",
        default="coreml",
        choices=["coreml", "onnx", "torchscript"],
        help="Export target (default: coreml)",
    )
    pe.add_argument(
        "--output",
        "-o",
        default=Path("models"),
        type=Path,
        help="Output directory (default: models/)",
    )
    pe.add_argument("--imgsz", type=int, default=640, help="Input size (default: 640)")
    pe.add_argument("--half", action="store_true", help="Export with fp16 weights")
    pe.add_argument(
        "--no-nms",
        action="store_true",
        help="Skip fusing NMS into the exported graph (not recommended for CoreML)",
    )
    pe.set_defaults(func=_cmd_export)

    # ── bench ─────────────────────────────────────────────────
    pb = sub.add_parser("bench", help="Benchmark detector throughput on a video clip.")
    pb.add_argument(
        "--weights", "-w", required=True, type=Path, help="Weights file (.pt, .mlpackage, .onnx)"
    )
    pb.add_argument(
        "--source", "-s", required=True, type=Path, help="Video clip to loop for timing"
    )
    pb.add_argument(
        "--device",
        default="mps",
        choices=["mps", "cpu", "cuda"],
        help="PyTorch device (ignored for CoreML)",
    )
    pb.add_argument("--imgsz", type=int, default=640)
    pb.add_argument("--conf", type=float, default=0.30)
    pb.add_argument("--frames", type=int, default=200, help="Timed frames (default: 200)")
    pb.add_argument("--warmup", type=int, default=10, help="Frames to skip before timing")
    pb.set_defaults(func=_cmd_bench)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    configure_logging(level="INFO", fmt="console")
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
