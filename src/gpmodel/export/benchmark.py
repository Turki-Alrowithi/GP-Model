"""Short, honest FPS/latency benchmark for a detector against a real clip.

Walks a fixed-length window of frames through the pipeline (detector
only, no tracker) and reports throughput + tail-latency percentiles.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

from gpmodel.detectors.yolo import YoloDetector
from gpmodel.sources.file import FileSource

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    weights: Path
    source: Path
    device: str
    imgsz: int
    n_frames: int
    avg_ms: float
    p50_ms: float
    p95_ms: float
    fps: float

    def pretty(self) -> str:
        return (
            f"\n  weights:   {self.weights}"
            f"\n  source:    {self.source}"
            f"\n  device:    {self.device}"
            f"\n  imgsz:     {self.imgsz}"
            f"\n  frames:    {self.n_frames}"
            f"\n  avg:       {self.avg_ms:6.2f} ms/frame"
            f"\n  p50:       {self.p50_ms:6.2f} ms"
            f"\n  p95:       {self.p95_ms:6.2f} ms"
            f"\n  fps:       {self.fps:6.1f}\n"
        )


def benchmark(
    weights: str | Path,
    source: str | Path,
    device: str = "mps",
    imgsz: int = 640,
    conf: float = 0.30,
    n_frames: int = 200,
    warmup: int = 10,
) -> BenchmarkResult:
    """Run `n_frames` through the detector, skipping the first `warmup` in timings."""
    det = YoloDetector(weights=weights, device=device, imgsz=imgsz, conf=conf)
    det.warmup()

    latencies: list[float] = []
    with FileSource(source, loop=True) as src:
        iterator = src.frames()
        for _ in range(warmup):
            frame = next(iterator)
            det.detect(frame)

        for _ in range(n_frames):
            frame = next(iterator)
            t0 = time.perf_counter()
            det.detect(frame)
            latencies.append((time.perf_counter() - t0) * 1000.0)

    latencies.sort()
    avg = mean(latencies)
    p50 = latencies[len(latencies) // 2]
    p95 = latencies[int(len(latencies) * 0.95)]
    fps = 1000.0 / avg if avg > 0 else 0.0

    return BenchmarkResult(
        weights=Path(weights),
        source=Path(source),
        device=device,
        imgsz=imgsz,
        n_frames=n_frames,
        avg_ms=avg,
        p50_ms=p50,
        p95_ms=p95,
        fps=fps,
    )
