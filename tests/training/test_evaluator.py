"""Tests for the evaluator wrapper — Ultralytics fully mocked."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from gpmodel.training.evaluator import evaluate


@patch("ultralytics.YOLO")
def test_evaluate_extracts_metrics(yolo_cls: MagicMock, tmp_path: Path) -> None:
    weights = tmp_path / "best.pt"
    weights.touch()
    data = tmp_path / "d.yaml"
    data.touch()

    model = MagicMock()
    metrics = MagicMock()
    metrics.box = MagicMock(map50=0.812, map=0.574, mp=0.78, mr=0.65)
    model.val.return_value = metrics
    yolo_cls.return_value = model

    r = evaluate(
        weights=weights, data=data, imgsz=512, batch=8, device="cpu", split="val"
    )

    assert r.map50 == pytest.approx(0.812)
    assert r.map50_95 == pytest.approx(0.574)
    assert r.precision == pytest.approx(0.78)
    assert r.recall == pytest.approx(0.65)
    assert r.imgsz == 512
    assert "mAP@50" in r.pretty()


def test_evaluate_raises_on_missing_weights(tmp_path: Path) -> None:
    d = tmp_path / "d.yaml"
    d.touch()
    with pytest.raises(FileNotFoundError):
        evaluate(weights=tmp_path / "nope.pt", data=d)


def test_evaluate_raises_on_missing_data(tmp_path: Path) -> None:
    w = tmp_path / "w.pt"
    w.touch()
    with pytest.raises(FileNotFoundError):
        evaluate(weights=w, data=tmp_path / "nope.yaml")


@patch("ultralytics.YOLO")
def test_evaluate_forwards_split(yolo_cls: MagicMock, tmp_path: Path) -> None:
    w = tmp_path / "w.pt"
    w.touch()
    d = tmp_path / "d.yaml"
    d.touch()

    model = MagicMock()
    metrics = MagicMock()
    metrics.box = MagicMock(map50=0.0, map=0.0, mp=0.0, mr=0.0)
    model.val.return_value = metrics
    yolo_cls.return_value = model

    evaluate(weights=w, data=d, split="test")
    assert model.val.call_args.kwargs["split"] == "test"
