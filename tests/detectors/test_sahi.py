"""Tests for SahiYoloDetector — SAHI is mocked so tests are offline."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from gpmodel.core.interfaces import Detector
from gpmodel.core.types import Frame


class _FakeScore:
    def __init__(self, value: float) -> None:
        self.value = value


class _FakeCategory:
    def __init__(self, cid: int) -> None:
        self.id = cid


class _FakeBBox:
    def __init__(self, xyxy: tuple[float, float, float, float]) -> None:
        self._xyxy = xyxy

    def to_xyxy(self) -> tuple[float, float, float, float]:
        return self._xyxy


class _FakePrediction:
    def __init__(self, xyxy: tuple[float, float, float, float], score: float, cid: int) -> None:
        self.bbox = _FakeBBox(xyxy)
        self.score = _FakeScore(score)
        self.category = _FakeCategory(cid)


@pytest.fixture
def fake_frame() -> Frame:
    img = np.zeros((720, 1280, 3), dtype=np.uint8)
    return Frame(stream_id="cam-1", frame_id=1, timestamp=datetime.now(UTC), image=img)


@patch("sahi.predict.get_sliced_prediction")
@patch("sahi.AutoDetectionModel")
def test_loads_model_and_exposes_names(auto_cls: MagicMock, _predict: MagicMock) -> None:
    from gpmodel.detectors.sahi import SahiYoloDetector

    model = MagicMock()
    model.category_mapping = {"0": "person", "2": "car"}
    auto_cls.from_pretrained.return_value = model

    det = SahiYoloDetector(weights="yolo11s.pt", device="cpu", slice_height=320, slice_width=320)

    auto_cls.from_pretrained.assert_called_once()
    kwargs = auto_cls.from_pretrained.call_args.kwargs
    assert kwargs["model_type"] == "ultralytics"
    assert kwargs["device"] == "cpu"
    assert det.class_names == {0: "person", 2: "car"}
    assert isinstance(det, Detector)


@patch("sahi.predict.get_sliced_prediction")
@patch("sahi.AutoDetectionModel")
def test_detect_returns_domain_objects(
    auto_cls: MagicMock, predict: MagicMock, fake_frame: Frame
) -> None:
    from gpmodel.detectors.sahi import SahiYoloDetector

    model = MagicMock()
    model.category_mapping = {"0": "person", "2": "car"}
    auto_cls.from_pretrained.return_value = model

    result = MagicMock()
    result.object_prediction_list = [
        _FakePrediction((10.0, 20.0, 110.0, 220.0), 0.92, 0),
        _FakePrediction((50.0, 60.0, 150.0, 260.0), 0.77, 2),
    ]
    predict.return_value = result

    det = SahiYoloDetector(weights="yolo11s.pt")
    out = det.detect(fake_frame)

    assert len(out) == 2
    assert out[0].class_name == "person"
    assert out[0].confidence == pytest.approx(0.92)
    assert out[0].bbox.as_xyxy() == (10.0, 20.0, 110.0, 220.0)
    assert out[1].class_name == "car"


@patch("sahi.predict.get_sliced_prediction")
@patch("sahi.AutoDetectionModel")
def test_detect_respects_class_filter(
    auto_cls: MagicMock, predict: MagicMock, fake_frame: Frame
) -> None:
    from gpmodel.detectors.sahi import SahiYoloDetector

    model = MagicMock()
    model.category_mapping = {"0": "person", "2": "car"}
    auto_cls.from_pretrained.return_value = model

    result = MagicMock()
    result.object_prediction_list = [
        _FakePrediction((0, 0, 10, 10), 0.9, 0),
        _FakePrediction((0, 0, 10, 10), 0.9, 2),
    ]
    predict.return_value = result

    det = SahiYoloDetector(weights="yolo11s.pt", classes=[0])  # persons only
    out = det.detect(fake_frame)

    assert len(out) == 1
    assert out[0].class_id == 0


@patch("sahi.predict.get_sliced_prediction")
@patch("sahi.AutoDetectionModel")
def test_predict_forwards_slice_and_overlap_params(
    auto_cls: MagicMock, predict: MagicMock, fake_frame: Frame
) -> None:
    from gpmodel.detectors.sahi import SahiYoloDetector

    model = MagicMock()
    model.category_mapping = {}
    auto_cls.from_pretrained.return_value = model
    predict.return_value = MagicMock(object_prediction_list=[])

    det = SahiYoloDetector(
        weights="yolo11s.pt",
        slice_height=1024,
        slice_width=1024,
        overlap_height_ratio=0.3,
        overlap_width_ratio=0.3,
        postprocess_type="NMS",
        postprocess_match_metric="IOU",
        postprocess_match_threshold=0.6,
    )
    det.detect(fake_frame)

    kwargs = predict.call_args.kwargs
    assert kwargs["slice_height"] == 1024
    assert kwargs["slice_width"] == 1024
    assert kwargs["overlap_height_ratio"] == pytest.approx(0.3)
    assert kwargs["overlap_width_ratio"] == pytest.approx(0.3)
    assert kwargs["postprocess_type"] == "NMS"
    assert kwargs["postprocess_match_metric"] == "IOU"
    assert kwargs["postprocess_match_threshold"] == 0.6


@patch("sahi.predict.get_sliced_prediction")
@patch("sahi.AutoDetectionModel")
def test_warmup_runs_one_inference(auto_cls: MagicMock, predict: MagicMock) -> None:
    from gpmodel.detectors.sahi import SahiYoloDetector

    model = MagicMock()
    model.category_mapping = {}
    auto_cls.from_pretrained.return_value = model
    predict.return_value = MagicMock(object_prediction_list=[])

    det = SahiYoloDetector(weights="yolo11s.pt", slice_height=256, slice_width=256)
    det.warmup()

    assert predict.called
    dummy_img = predict.call_args.kwargs["image"]
    # Warmup image is 2x the slice size so the detector actually has to slice.
    assert dummy_img.shape == (512, 512, 3)
