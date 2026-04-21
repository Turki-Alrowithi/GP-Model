"""Tests for dataset merging."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from gpmodel.training.merge import ClassMap, load_class_map, merge_datasets


def _build_source_dataset(
    root: Path,
    name: str,
    class_names: list[str],
    splits: dict[str, list[tuple[str, list[tuple[int, float, float, float, float]]]]],
) -> Path:
    """Create a small fake YOLO dataset under `root/<name>/`.

    `splits` maps split name to a list of (image_basename, [(class_id, cx, cy, w, h), ...]).
    """
    ds_dir = root / name
    data_paths: dict[str, str] = {}
    for split, entries in splits.items():
        img_dir = ds_dir / "images" / split
        lbl_dir = ds_dir / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        for stem, boxes in entries:
            (img_dir / f"{stem}.jpg").write_bytes(b"fake-image")
            label_lines = [
                f"{cid} {cx:.3f} {cy:.3f} {w:.3f} {h:.3f}" for (cid, cx, cy, w, h) in boxes
            ]
            (lbl_dir / f"{stem}.txt").write_text("\n".join(label_lines) + "\n")
        data_paths[split] = f"images/{split}"

    yaml_path = root / f"{name}.yaml"
    yaml_body = {
        "path": f"./{name}",
        "nc": len(class_names),
        "names": class_names,
        **data_paths,
    }
    yaml_path.write_text(yaml.safe_dump(yaml_body, sort_keys=False))
    return yaml_path


def _read_label(label_path: Path) -> list[list[str]]:
    return [line.split() for line in label_path.read_text().splitlines() if line.strip()]


def test_merge_remaps_class_ids_and_writes_data_yaml(tmp_path: Path) -> None:
    sohas = _build_source_dataset(
        tmp_path,
        "sohas",
        class_names=["pistol", "knife", "phone"],
        splits={
            "train": [
                ("img1", [(0, 0.5, 0.5, 0.2, 0.2), (1, 0.3, 0.4, 0.1, 0.1)]),
                # phone-only image — must be dropped entirely.
                ("img2", [(2, 0.5, 0.5, 0.2, 0.2)]),
            ],
            "val": [("img3", [(1, 0.1, 0.1, 0.1, 0.1)])],
        },
    )

    class_map = ClassMap(
        targets=["firearm", "knife"],
        mapping={"pistol": "firearm", "knife": "knife"},
    )

    output = tmp_path / "merged"
    stats = merge_datasets([sohas], class_map, output)

    # Dropped the phone-only image entirely.
    assert stats.images_linked == 2
    assert stats.labels_written == 2
    assert "phone" in stats.dropped_classes

    # Check remapped labels.
    train_label = output / "labels" / "train" / "sohas__img1.txt"
    lines = _read_label(train_label)
    # pistol (0) → firearm (0); knife (1) → knife (1).
    class_ids = sorted(line[0] for line in lines)
    assert class_ids == ["0", "1"]

    val_label = output / "labels" / "val" / "sohas__img3.txt"
    assert _read_label(val_label)[0][0] == "1"  # knife → 1

    # Generated data.yaml is usable as-is.
    merged_yaml = yaml.safe_load((output / "data.yaml").read_text())
    assert merged_yaml["names"] == ["firearm", "knife"]
    assert merged_yaml["nc"] == 2
    assert merged_yaml["train"] == "images/train"
    assert merged_yaml["val"] == "images/val"
    assert "test" not in merged_yaml  # source didn't have one

    # Symlinks point at the source images.
    train_img = output / "images" / "train" / "sohas__img1.jpg"
    assert train_img.is_symlink()
    assert train_img.resolve() == (tmp_path / "sohas" / "images" / "train" / "img1.jpg").resolve()


def test_merge_combines_two_sources_with_shared_target(tmp_path: Path) -> None:
    sohas = _build_source_dataset(
        tmp_path,
        "sohas",
        class_names=["pistol"],
        splits={"train": [("s1", [(0, 0.5, 0.5, 0.2, 0.2)])]},
    )
    olmos = _build_source_dataset(
        tmp_path,
        "olmos",
        class_names=["gun"],
        splits={"train": [("o1", [(0, 0.4, 0.4, 0.2, 0.2)])]},
    )

    class_map = ClassMap(
        targets=["firearm"],
        mapping={"pistol": "firearm", "gun": "firearm"},
    )
    output = tmp_path / "merged"
    stats = merge_datasets([sohas, olmos], class_map, output)

    assert stats.images_linked == 2
    images = sorted(p.name for p in (output / "images" / "train").iterdir())
    assert images == ["olmos__o1.jpg", "sohas__s1.jpg"]


def test_merge_drops_unmapped_lines(tmp_path: Path) -> None:
    ds = _build_source_dataset(
        tmp_path,
        "ds",
        class_names=["pistol", "phone"],
        splits={
            "train": [
                # Mixed image: pistol kept, phone dropped.
                ("img", [(0, 0.5, 0.5, 0.2, 0.2), (1, 0.1, 0.1, 0.1, 0.1)]),
            ]
        },
    )
    class_map = ClassMap(targets=["firearm"], mapping={"pistol": "firearm"})

    output = tmp_path / "merged"
    stats = merge_datasets([ds], class_map, output)

    assert stats.images_linked == 1
    label_lines = _read_label(output / "labels" / "train" / "ds__img.txt")
    assert len(label_lines) == 1
    assert label_lines[0][0] == "0"
    assert "phone" in stats.dropped_classes


def test_copy_mode_makes_regular_files(tmp_path: Path) -> None:
    ds = _build_source_dataset(
        tmp_path,
        "ds",
        class_names=["pistol"],
        splits={"train": [("img", [(0, 0.5, 0.5, 0.2, 0.2)])]},
    )
    class_map = ClassMap(targets=["firearm"], mapping={"pistol": "firearm"})
    output = tmp_path / "merged"
    merge_datasets([ds], class_map, output, copy_images=True)

    img = output / "images" / "train" / "ds__img.jpg"
    assert img.is_file()
    assert not img.is_symlink()


def test_load_class_map_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "cm.yaml"
    path.write_text("targets: [firearm, knife]\nmapping:\n  pistol: firearm\n  knife: knife\n")
    cm = load_class_map(path)
    assert cm.targets == ["firearm", "knife"]
    assert cm.target_id("pistol") == 0
    assert cm.target_id("knife") == 1
    assert cm.target_id("phone") is None


def test_class_map_raises_on_target_not_in_vocab() -> None:
    cm = ClassMap(targets=["firearm"], mapping={"pistol": "missing"})
    with pytest.raises(ValueError, match="isn't in `targets`"):
        cm.target_id("pistol")


def test_load_class_map_requires_targets(tmp_path: Path) -> None:
    path = tmp_path / "cm.yaml"
    path.write_text("mapping:\n  pistol: firearm\n")
    with pytest.raises(ValueError, match="'targets' is required"):
        load_class_map(path)


def test_missing_source_raises(tmp_path: Path) -> None:
    class_map = ClassMap(targets=["firearm"], mapping={})
    with pytest.raises(FileNotFoundError):
        merge_datasets([tmp_path / "nope.yaml"], class_map, tmp_path / "out")
