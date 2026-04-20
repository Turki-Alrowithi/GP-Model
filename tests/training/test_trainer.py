"""Tests for the training wrapper — Ultralytics fully mocked."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from gpmodel.training.trainer import train


@patch("ultralytics.YOLO")
def test_train_builds_overrides_and_returns_best_weights(
    yolo_cls: MagicMock, tmp_path: Path
) -> None:
    weights = tmp_path / "yolo11s.pt"
    weights.touch()
    data = tmp_path / "weapons.yaml"
    data.write_text("path: .\ntrain: images/train\nval: images/val\nnc: 1\nnames: [knife]\n")

    run_dir = tmp_path / "runs" / "train" / "exp"
    (run_dir / "weights").mkdir(parents=True)
    (run_dir / "weights" / "best.pt").write_bytes(b"")
    (run_dir / "weights" / "last.pt").write_bytes(b"")

    model = MagicMock()
    results = MagicMock()
    results.save_dir = str(run_dir)
    model.train.return_value = results
    yolo_cls.return_value = model

    result = train(
        base_weights=weights,
        data=data,
        epochs=3,
        imgsz=320,
        batch=8,
        device="cpu",
        project=tmp_path / "runs" / "train",
        name="exp",
    )

    kwargs = model.train.call_args.kwargs
    assert kwargs["data"] == str(data)
    assert kwargs["epochs"] == 3
    assert kwargs["imgsz"] == 320
    assert kwargs["batch"] == 8
    assert kwargs["device"] == "cpu"
    assert kwargs["name"] == "exp"
    assert "lr0" not in kwargs  # not passed when None

    assert result.best_weights == run_dir / "weights" / "best.pt"
    assert result.last_weights == run_dir / "weights" / "last.pt"
    assert result.run_dir == run_dir
    assert result.epochs == 3


def test_train_raises_on_missing_weights(tmp_path: Path) -> None:
    data = tmp_path / "d.yaml"
    data.touch()
    with pytest.raises(FileNotFoundError):
        train(base_weights=tmp_path / "nope.pt", data=data)


def test_train_raises_on_missing_data(tmp_path: Path) -> None:
    weights = tmp_path / "yolo11s.pt"
    weights.touch()
    with pytest.raises(FileNotFoundError):
        train(base_weights=weights, data=tmp_path / "nope.yaml")


@patch("ultralytics.YOLO")
def test_train_forwards_optional_lr0(yolo_cls: MagicMock, tmp_path: Path) -> None:
    weights = tmp_path / "w.pt"
    weights.touch()
    data = tmp_path / "d.yaml"
    data.touch()
    run_dir = tmp_path / "runs" / "train" / "exp"
    (run_dir / "weights").mkdir(parents=True)
    (run_dir / "weights" / "best.pt").write_bytes(b"")
    (run_dir / "weights" / "last.pt").write_bytes(b"")

    model = MagicMock()
    model.train.return_value = MagicMock(save_dir=str(run_dir))
    yolo_cls.return_value = model

    train(
        base_weights=weights, data=data, lr0=0.001, project=tmp_path / "runs" / "train", name="exp"
    )

    assert model.train.call_args.kwargs["lr0"] == 0.001


@patch("ultralytics.YOLO")
def test_train_raises_when_best_missing(yolo_cls: MagicMock, tmp_path: Path) -> None:
    weights = tmp_path / "w.pt"
    weights.touch()
    data = tmp_path / "d.yaml"
    data.touch()
    run_dir = tmp_path / "runs" / "train" / "exp"
    run_dir.mkdir(parents=True)
    # Deliberately don't create weights/best.pt

    model = MagicMock()
    model.train.return_value = MagicMock(save_dir=str(run_dir))
    yolo_cls.return_value = model

    with pytest.raises(RuntimeError, match=r"best\.pt missing"):
        train(base_weights=weights, data=data, project=tmp_path / "runs" / "train", name="exp")
