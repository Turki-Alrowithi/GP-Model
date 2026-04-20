"""Tests for the benchmark helper — YoloDetector is mocked so tests are offline."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from gpmodel.export.benchmark import benchmark


@patch("ultralytics.YOLO")
def test_benchmark_returns_plausible_stats(yolo_cls: MagicMock) -> None:
    # Arrange the Ultralytics mock.
    model = MagicMock()
    model.names = {0: "person"}
    # A minimal fake Results object with zero detections is enough.
    boxes = MagicMock()
    boxes.__len__.return_value = 0
    result = MagicMock()
    result.boxes = boxes
    model.predict.return_value = [result]
    yolo_cls.return_value = model

    sample = Path(__file__).resolve().parents[2] / "assets" / "samples" / "drone_01.mp4"
    if not sample.exists():
        import pytest

        pytest.skip(f"missing sample clip: {sample}")

    res = benchmark(
        weights="yolo11s.pt",
        source=sample,
        device="cpu",
        imgsz=320,
        n_frames=20,
        warmup=2,
    )

    assert res.n_frames == 20
    assert res.avg_ms > 0
    assert res.p50_ms > 0
    assert res.p95_ms >= res.p50_ms
    assert res.fps > 0
    assert res.device == "cpu"
    assert "fps" in res.pretty()
