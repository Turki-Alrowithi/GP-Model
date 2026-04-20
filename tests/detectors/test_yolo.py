"""Tests for YoloDetector.

Unit tests mock the Ultralytics YOLO class — fast, offline, deterministic.
An integration test that runs real inference is marked `integration`
and downloads weights on demand.
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from gpmodel.core.interfaces import Detector
from gpmodel.core.types import Frame


class FakeBoxes:
    """Mimics the subset of ultralytics Results.boxes that YoloDetector reads."""

    def __init__(self, xyxy: np.ndarray, conf: np.ndarray, cls: np.ndarray) -> None:
        self.xyxy = _Tensor(xyxy)
        self.conf = _Tensor(conf)
        self.cls = _Tensor(cls)

    def __len__(self) -> int:
        return len(self.xyxy.numpy())


class _Tensor:
    def __init__(self, arr: np.ndarray) -> None:
        self._arr = arr

    def cpu(self) -> _Tensor:
        return self

    def numpy(self) -> np.ndarray:
        return self._arr


class FakeResult:
    def __init__(self, boxes: FakeBoxes) -> None:
        self.boxes = boxes


@pytest.fixture
def fake_frame() -> Frame:
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    return Frame(stream_id="cam-1", frame_id=1, timestamp=datetime.now(UTC), image=img)


@patch("ultralytics.YOLO")
def test_loads_model_and_exposes_names(yolo_cls: MagicMock) -> None:
    from gpmodel.detectors.yolo import YoloDetector

    model = MagicMock()
    model.names = {0: "person", 2: "car"}
    yolo_cls.return_value = model

    det = YoloDetector(weights="yolo11s.pt", device="mps", conf=0.4)

    yolo_cls.assert_called_once_with("yolo11s.pt")
    assert det.class_names == {0: "person", 2: "car"}
    assert det.device == "mps"
    assert det.conf == 0.4
    assert isinstance(det, Detector)


@patch("ultralytics.YOLO")
def test_detect_returns_domain_objects(yolo_cls: MagicMock, fake_frame: Frame) -> None:
    from gpmodel.detectors.yolo import YoloDetector

    model = MagicMock()
    model.names = {0: "person", 2: "car"}
    boxes = FakeBoxes(
        xyxy=np.array([[10.0, 20.0, 110.0, 220.0], [50.0, 60.0, 150.0, 260.0]]),
        conf=np.array([0.92, 0.77]),
        cls=np.array([0, 2]),
    )
    model.predict.return_value = [FakeResult(boxes)]
    yolo_cls.return_value = model

    det = YoloDetector(weights="yolo11s.pt")
    out = det.detect(fake_frame)

    assert len(out) == 2
    assert out[0].class_id == 0
    assert out[0].class_name == "person"
    assert out[0].confidence == pytest.approx(0.92)
    assert out[0].bbox.as_xyxy() == (10.0, 20.0, 110.0, 220.0)
    assert out[1].class_name == "car"


@patch("ultralytics.YOLO")
def test_detect_handles_empty_results(yolo_cls: MagicMock, fake_frame: Frame) -> None:
    from gpmodel.detectors.yolo import YoloDetector

    model = MagicMock()
    model.names = {0: "person"}
    model.predict.return_value = []
    yolo_cls.return_value = model

    det = YoloDetector(weights="yolo11s.pt")
    assert det.detect(fake_frame) == []


@patch("ultralytics.YOLO")
def test_passes_inference_kwargs(yolo_cls: MagicMock, fake_frame: Frame) -> None:
    from gpmodel.detectors.yolo import YoloDetector

    model = MagicMock()
    model.names = {}
    empty_boxes = FakeBoxes(xyxy=np.zeros((0, 4)), conf=np.zeros(0), cls=np.zeros(0))
    model.predict.return_value = [FakeResult(empty_boxes)]
    yolo_cls.return_value = model

    det = YoloDetector(
        weights="yolo11s.pt",
        device="cpu",
        imgsz=960,
        conf=0.5,
        iou=0.6,
        classes=[0, 2],
        half=True,
    )
    det.detect(fake_frame)

    kwargs = model.predict.call_args.kwargs
    assert kwargs["device"] == "cpu"
    assert kwargs["imgsz"] == 960
    assert kwargs["conf"] == 0.5
    assert kwargs["iou"] == 0.6
    assert kwargs["classes"] == [0, 2]
    assert kwargs["half"] is True
    assert kwargs["verbose"] is False


@patch("ultralytics.YOLO")
def test_warmup_calls_predict(yolo_cls: MagicMock) -> None:
    from gpmodel.detectors.yolo import YoloDetector

    model = MagicMock()
    model.names = {}
    empty_boxes = FakeBoxes(xyxy=np.zeros((0, 4)), conf=np.zeros(0), cls=np.zeros(0))
    model.predict.return_value = [FakeResult(empty_boxes)]
    yolo_cls.return_value = model

    det = YoloDetector(weights="yolo11s.pt", imgsz=320)
    det.warmup()

    assert model.predict.called
    source = model.predict.call_args.kwargs["source"]
    assert source.shape == (320, 320, 3)
