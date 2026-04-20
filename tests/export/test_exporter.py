"""Tests for export_model — Ultralytics is mocked so tests are offline."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from gpmodel.export.exporter import export_model


@patch("ultralytics.YOLO")
def test_export_moves_artifact_to_models_dir(yolo_cls: MagicMock, tmp_path: Path) -> None:
    # Arrange: fake weights and a fake artefact Ultralytics "produced".
    weights = tmp_path / "yolo11s.pt"
    weights.touch()
    produced = tmp_path / "yolo11s.mlpackage"
    produced.mkdir()  # CoreML exports are directories

    model = MagicMock()
    model.export.return_value = str(produced)
    yolo_cls.return_value = model

    out_dir = tmp_path / "models"
    result = export_model(weights=weights, fmt="coreml", output_dir=out_dir, imgsz=640)

    # Assert: moved into the requested dir with the expected suffix.
    expected = out_dir / "yolo11s.mlpackage"
    assert result.path == expected
    assert expected.exists()
    assert not produced.exists()  # original was moved, not copied
    assert result.format == "coreml"
    assert result.imgsz == 640


@patch("ultralytics.YOLO")
def test_export_overwrites_existing_artifact(yolo_cls: MagicMock, tmp_path: Path) -> None:
    weights = tmp_path / "yolo11s.pt"
    weights.touch()
    produced = tmp_path / "yolo11s.onnx"
    produced.write_bytes(b"new")

    # Pre-existing stale export.
    models = tmp_path / "models"
    models.mkdir()
    (models / "yolo11s.onnx").write_bytes(b"old")

    model = MagicMock()
    model.export.return_value = str(produced)
    yolo_cls.return_value = model

    result = export_model(weights=weights, fmt="onnx", output_dir=models)
    assert result.path.read_bytes() == b"new"


def test_export_raises_on_missing_weights(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        export_model(weights=tmp_path / "missing.pt", fmt="coreml")


@patch("ultralytics.YOLO")
def test_export_raises_when_artifact_missing(yolo_cls: MagicMock, tmp_path: Path) -> None:
    weights = tmp_path / "yolo11s.pt"
    weights.touch()
    model = MagicMock()
    model.export.return_value = str(tmp_path / "does_not_exist.onnx")
    yolo_cls.return_value = model

    with pytest.raises(RuntimeError, match="output is missing"):
        export_model(weights=weights, fmt="onnx", output_dir=tmp_path / "models")


@patch("ultralytics.YOLO")
def test_export_forwards_format_args(yolo_cls: MagicMock, tmp_path: Path) -> None:
    weights = tmp_path / "yolo11s.pt"
    weights.touch()
    produced = tmp_path / "yolo11s.torchscript"
    produced.touch()
    model = MagicMock()
    model.export.return_value = str(produced)
    yolo_cls.return_value = model

    export_model(
        weights=weights,
        fmt="torchscript",
        output_dir=tmp_path / "models",
        imgsz=1280,
        half=True,
        nms=False,
    )

    kwargs = model.export.call_args.kwargs
    assert kwargs["format"] == "torchscript"
    assert kwargs["imgsz"] == 1280
    assert kwargs["half"] is True
    assert kwargs["nms"] is False
