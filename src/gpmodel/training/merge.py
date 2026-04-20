"""Merge multiple YOLO-format datasets into a single dataset with remapped classes.

Typical use: combine Sohas + Olmos + Objects365-weapons into one
unified `weapons` dataset whose class vocabulary matches the rule
engine's `WeaponRuleConfig.classes`.

Inputs:

- A list of source Ultralytics-format dataset YAMLs.
- A mapping YAML that declares the merged vocabulary and maps each
  source class name to one of the target names. Source classes not
  listed in the mapping are silently dropped — an easy way to filter
  noise classes (e.g. "phone" in Sohas, which we don't want to train
  a firearm model to "detect").

Outputs:

- `output/images/{train,val,test}/…` images (symlinked by default).
- `output/labels/{train,val,test}/…` YOLO-format labels with the
  target class ids.
- `output/data.yaml` ready to pass straight to `apps/train.py`.
"""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

_SPLITS = ("train", "val", "test")


@dataclass(frozen=True, slots=True)
class ClassMap:
    """Target vocabulary plus per-source-name → target-name mapping."""

    targets: list[str]
    mapping: dict[str, str]

    def target_id(self, source_name: str) -> int | None:
        target = self.mapping.get(source_name)
        if target is None:
            return None
        try:
            return self.targets.index(target)
        except ValueError:
            raise ValueError(
                f"Mapping points '{source_name}' → '{target}', which isn't in `targets`."
            ) from None


@dataclass(frozen=True, slots=True)
class MergeStats:
    images_linked: int
    labels_written: int
    dropped_classes: set[str]


def load_class_map(path: str | Path) -> ClassMap:
    data: dict[str, Any] = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    targets = list(data.get("targets", []))
    mapping = dict(data.get("mapping", {}))
    if not targets:
        raise ValueError(f"{path}: 'targets' is required and must list the merged class names.")
    return ClassMap(targets=targets, mapping=mapping)


def merge_datasets(
    source_yamls: list[str | Path],
    class_map: ClassMap,
    output_dir: str | Path,
    copy_images: bool = False,
) -> MergeStats:
    """Merge `source_yamls` into `output_dir` using `class_map`.

    Returns a summary of what happened so the CLI can print it.
    """
    out_root = Path(output_dir).resolve()
    (out_root / "images").mkdir(parents=True, exist_ok=True)
    (out_root / "labels").mkdir(parents=True, exist_ok=True)

    images_linked = 0
    labels_written = 0
    dropped: set[str] = set()

    for src_yaml in source_yamls:
        src_path = Path(src_yaml).resolve()
        if not src_path.exists():
            raise FileNotFoundError(f"Source dataset YAML not found: {src_path}")
        logger.info("Merging %s", src_path)
        stats = _merge_one(src_path, class_map, out_root, copy_images)
        images_linked += stats.images_linked
        labels_written += stats.labels_written
        dropped |= stats.dropped_classes

    data_yaml_path = _write_merged_yaml(out_root, class_map)
    logger.info(
        "Merge complete: %d images, %d labels, %d dropped class names. data.yaml=%s",
        images_linked, labels_written, len(dropped), data_yaml_path,
    )
    return MergeStats(
        images_linked=images_linked,
        labels_written=labels_written,
        dropped_classes=dropped,
    )


# ── Internals ────────────────────────────────────────────
def _merge_one(
    src_yaml: Path, class_map: ClassMap, out_root: Path, copy_images: bool
) -> MergeStats:
    data = yaml.safe_load(src_yaml.read_text(encoding="utf-8")) or {}
    src_root = (src_yaml.parent / data.get("path", ".")).resolve()
    names = data.get("names", [])
    if not isinstance(names, list):
        # Ultralytics also accepts dict[int, str]; normalise to list by index.
        names = [names[i] for i in sorted(names)]

    stats = MergeStats(images_linked=0, labels_written=0, dropped_classes=set())

    for split in _SPLITS:
        split_images_rel = data.get(split)
        if not split_images_rel:
            continue
        src_images_dir = (src_root / split_images_rel).resolve()
        if not src_images_dir.exists():
            logger.warning("Split '%s' listed but missing on disk: %s", split, src_images_dir)
            continue
        # YOLO convention: labels live next to images under a sibling `labels/<split>/`.
        src_labels_dir = _labels_dir_for(src_root, src_images_dir, split)
        out_images_dir = out_root / "images" / split
        out_labels_dir = out_root / "labels" / split
        out_images_dir.mkdir(parents=True, exist_ok=True)
        out_labels_dir.mkdir(parents=True, exist_ok=True)

        s = _merge_split(
            src_images_dir,
            src_labels_dir,
            out_images_dir,
            out_labels_dir,
            names,
            class_map,
            copy_images,
            prefix=src_yaml.stem,
        )
        stats = MergeStats(
            images_linked=stats.images_linked + s.images_linked,
            labels_written=stats.labels_written + s.labels_written,
            dropped_classes=stats.dropped_classes | s.dropped_classes,
        )
    return stats


def _labels_dir_for(src_root: Path, images_dir: Path, split: str) -> Path:
    # Try the standard sibling layout first (images/train/ ↔ labels/train/).
    try:
        rel = images_dir.relative_to(src_root)
        parts = list(rel.parts)
        if parts and parts[0] == "images":
            parts[0] = "labels"
            return src_root / Path(*parts)
    except ValueError:
        pass
    return src_root / "labels" / split


def _merge_split(
    src_images_dir: Path,
    src_labels_dir: Path,
    out_images_dir: Path,
    out_labels_dir: Path,
    source_names: list[str],
    class_map: ClassMap,
    copy_images: bool,
    prefix: str,
) -> MergeStats:
    images_linked = 0
    labels_written = 0
    dropped: set[str] = set()

    for img in sorted(src_images_dir.rglob("*")):
        if not img.is_file() or img.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
            continue
        label = src_labels_dir / f"{img.stem}.txt"
        if not label.exists():
            continue

        # Remap labels — drop any lines whose class isn't in the mapping.
        kept_lines: list[str] = []
        for line in label.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                src_cid = int(parts[0])
            except ValueError:
                continue
            if src_cid < 0 or src_cid >= len(source_names):
                continue
            src_name = source_names[src_cid]
            target_cid = class_map.target_id(src_name)
            if target_cid is None:
                dropped.add(src_name)
                continue
            parts[0] = str(target_cid)
            kept_lines.append(" ".join(parts))

        if not kept_lines:
            continue  # no usable labels in this image

        # Unique name so multi-source merges don't collide.
        out_img = out_images_dir / f"{prefix}__{img.name}"
        out_lbl = out_labels_dir / f"{prefix}__{img.stem}.txt"

        _place_image(img, out_img, copy_images)
        out_lbl.write_text("\n".join(kept_lines) + "\n", encoding="utf-8")

        images_linked += 1
        labels_written += 1

    return MergeStats(
        images_linked=images_linked, labels_written=labels_written, dropped_classes=dropped
    )


def _place_image(src: Path, dst: Path, copy: bool) -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if copy:
        shutil.copy2(src, dst)
    else:
        # Absolute symlink so it keeps working regardless of CWD.
        dst.symlink_to(src.resolve())


def _write_merged_yaml(out_root: Path, class_map: ClassMap) -> Path:
    existing_splits = {s: (out_root / "images" / s) for s in _SPLITS}
    existing_splits = {s: p for s, p in existing_splits.items() if p.exists()}
    yaml_body: dict[str, Any] = {
        "path": str(out_root),
        "nc": len(class_map.targets),
        "names": list(class_map.targets),
    }
    for split, path in existing_splits.items():
        yaml_body[split] = str(path.relative_to(out_root))

    out_yaml = out_root / "data.yaml"
    out_yaml.write_text(yaml.safe_dump(yaml_body, sort_keys=False), encoding="utf-8")
    return out_yaml
