#!/usr/bin/env python3
"""CLI: combine multiple YOLO-format datasets into one with remapped classes.

Example:

    uv run python apps/merge_datasets.py \\
        --sources datasets/sohas.yaml datasets/olmos.yaml \\
        --mapping datasets/class_map.yaml \\
        --output datasets/weapons_merged

The output directory will contain `images/{train,val,test}/` (as
symlinks back to the source images), `labels/{train,val,test}/`
(YOLO-format labels with remapped class ids), and a ready-to-use
`data.yaml` for training.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from gpmodel.telemetry.logging import configure_logging
from gpmodel.training.merge import load_class_map, merge_datasets


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="gpmodel-merge",
        description="Merge multiple YOLO-format datasets with class remapping.",
    )
    p.add_argument(
        "--sources",
        "-s",
        type=Path,
        nargs="+",
        required=True,
        help="One or more source dataset YAMLs",
    )
    p.add_argument(
        "--mapping",
        "-m",
        type=Path,
        required=True,
        help="Class-mapping YAML (see datasets/class_map.example.yaml)",
    )
    p.add_argument(
        "--output", "-o", type=Path, required=True, help="Output directory for the merged dataset"
    )
    p.add_argument(
        "--copy",
        action="store_true",
        help="Copy images instead of symlinking (disk-heavy; useful for Colab/Drive)",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_logging(level="INFO", fmt="console")

    class_map = load_class_map(args.mapping)
    stats = merge_datasets(
        source_yamls=list(args.sources),
        class_map=class_map,
        output_dir=args.output,
        copy_images=args.copy,
    )

    print("\nMerge complete.")
    print(f"  Output:         {args.output}")
    print(f"  data.yaml:      {args.output}/data.yaml")
    print(f"  Images:         {stats.images_linked}")
    print(f"  Labels:         {stats.labels_written}")
    print(f"  Target classes: {class_map.targets}")
    if stats.dropped_classes:
        print(f"  Dropped:        {sorted(stats.dropped_classes)}")
    print("\nTrain with:")
    print(f"  uv run python apps/train.py --data {args.output}/data.yaml --epochs 50")
    return 0


if __name__ == "__main__":
    sys.exit(main())
